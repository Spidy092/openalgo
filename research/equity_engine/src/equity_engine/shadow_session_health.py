"""Read-only shadow session health monitor V1 (no orders, no network).

Monitors Monday's read-only shadow runner without changing strategy behavior
and without any broker-order capability:

- runner started/stopped
- last quote received time
- quote age vs freshness threshold
- expected polling cadence vs actual
- feed gaps, reconnect boundaries, duplicate/out-of-order events
- session/CAS state
- decisions / theoretical trades counts
- evidence persistence status, checksums, report generation
- disk/write errors, operator kill switch, fail-closed state
- live_orders_called=false

Produces a deterministic machine-readable :class:`ShadowSessionHealthReport`
with one of ``HEALTHY``, ``DEGRADED_NO_TRADING``, ``FAILED_CLOSED``.

Deterministic gate rules (load-bearing, explicit):

- ``FAILED_CLOSED`` (never allow new theoretical trades):
  operator kill sentinel present, disk/write error recorded, persistence
  incomplete (missing required files), or checksum/report mismatch
  (evidence corruption).
- ``DEGRADED_NO_TRADING`` (no new theoretical trades until recovered):
  stale last quote, cadence breach, unresolved feed gap, unresolved
  reconnect boundary, recent duplicate/out-of-order burst beyond tolerance,
  CAS/post-continuous uncertainty, or session closed. Recovery requires
  ``required_recovery_polls`` consecutive ``ok`` normalized events after the
  last anomaly; gaps are never hidden or silently recovered.
- ``HEALTHY`` only when none of the above hold.

This module performs no notifications and opens no network connections. The
only network path in the shadow stack remains the existing read-only
market-data GET used by the runner.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

SCHEMA_VERSION = "shadow-session-health/v1"
RUNNER_TIMEZONE = "Asia/Kolkata"
KILL_SENTINEL = "KILL"
DISK_ERROR_SENTINEL = "DISK_ERROR"

REQUIRED_EVIDENCE_FILES = (
    "config.json",
    "market_events.jsonl",
    "decisions.jsonl",
    "trades.jsonl",
    "summary.json",
    "CHECKSUMS.sha256",
)


class HealthStatus(StrEnum):
    HEALTHY = "HEALTHY"
    DEGRADED_NO_TRADING = "DEGRADED_NO_TRADING"
    FAILED_CLOSED = "FAILED_CLOSED"


@dataclass(frozen=True)
class ShadowSessionHealthReport:
    schema_version: str
    session_id: str
    status: HealthStatus
    allow_new_theoretical_trades: bool
    runner_started: bool
    runner_stopped: bool
    killed: bool
    last_quote_received_at: str | None
    last_quote_age_seconds: float | None
    expected_cadence_seconds: float | None
    seconds_since_last_quote: float | None
    feed_gaps: int
    reconnect_boundaries: int
    duplicates: int
    out_of_order: int
    cas_auxiliary_events: int
    outside_continuous_events: int
    session_closed: bool
    decisions_count: int
    theoretical_trades_count: int
    persistence_ok: bool
    checksums_ok: bool | None
    reports_ok: bool | None
    disk_error: bool
    reasons: tuple[str, ...]
    live_orders_called: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {self.schema_version!r}")
        if self.live_orders_called:
            raise ValueError("health reports never call live orders")
        if self.status is HealthStatus.HEALTHY and not self.allow_new_theoretical_trades:
            raise ValueError("HEALTHY must allow new theoretical trades")
        if (
            self.status
            in (
                HealthStatus.DEGRADED_NO_TRADING,
                HealthStatus.FAILED_CLOSED,
            )
            and self.allow_new_theoretical_trades
        ):
            raise ValueError(f"{self.status} must not allow new theoretical trades")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "status": self.status.value,
            "allow_new_theoretical_trades": self.allow_new_theoretical_trades,
            "runner_started": self.runner_started,
            "runner_stopped": self.runner_stopped,
            "killed": self.killed,
            "last_quote_received_at": self.last_quote_received_at,
            "last_quote_age_seconds": self.last_quote_age_seconds,
            "expected_cadence_seconds": self.expected_cadence_seconds,
            "seconds_since_last_quote": self.seconds_since_last_quote,
            "feed_gaps": self.feed_gaps,
            "reconnect_boundaries": self.reconnect_boundaries,
            "duplicates": self.duplicates,
            "out_of_order": self.out_of_order,
            "cas_auxiliary_events": self.cas_auxiliary_events,
            "outside_continuous_events": self.outside_continuous_events,
            "session_closed": self.session_closed,
            "decisions_count": self.decisions_count,
            "theoretical_trades_count": self.theoretical_trades_count,
            "persistence_ok": self.persistence_ok,
            "checksums_ok": self.checksums_ok,
            "reports_ok": self.reports_ok,
            "disk_error": self.disk_error,
            "reasons": list(self.reasons),
            "live_orders_called": False,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


@dataclass(frozen=True)
class HealthThresholds:
    freshness_seconds: float
    cadence_multiplier: float = 1.5
    required_recovery_polls: int = 2
    max_duplicates_tolerated: int = 0
    max_out_of_order_tolerated: int = 0

    def __post_init__(self) -> None:
        if self.freshness_seconds <= 0:
            raise ValueError("freshness_seconds must be positive")
        if self.cadence_multiplier < 1:
            raise ValueError("cadence_multiplier must be >= 1")
        if self.required_recovery_polls < 1:
            raise ValueError("required_recovery_polls must be >= 1")
        if self.max_duplicates_tolerated < 0 or self.max_out_of_order_tolerated < 0:
            raise ValueError("tolerances cannot be negative")


def evaluate_session_health(
    *,
    session_id: str,
    runner_started: bool,
    runner_stopped: bool,
    killed: bool,
    disk_error: bool,
    persistence_ok: bool,
    checksums_ok: bool | None,
    reports_ok: bool | None,
    normalized_feed_statuses: tuple[str, ...],
    normalized_received_ats: tuple[str | None, ...],
    normalized_reconnects: tuple[bool, ...] = (),
    decision_reasons: tuple[str, ...] = (),
    decisions_count: int = 0,
    theoretical_trades_count: int = 0,
    session_closed: bool = False,
    expected_cadence_seconds: float | None = None,
    last_quote_age_seconds: float | None = None,
    now_iso: str | None = None,
    thresholds: HealthThresholds | None = None,
) -> ShadowSessionHealthReport:
    """Deterministically evaluate shadow session health (pure, read-only)."""
    limits = thresholds or HealthThresholds(freshness_seconds=60.0)
    reasons: list[str] = []

    feed_gaps = sum(1 for status in normalized_feed_statuses if status == "feed_gap")
    feed_gaps += sum(1 for r in decision_reasons if r == "feed_gap_no_trade")
    reconnects = sum(1 for flag in normalized_reconnects if flag)
    reconnects += sum(1 for status in normalized_feed_statuses if status == "poll_failure")
    duplicates = sum(1 for r in decision_reasons if r == "duplicate_event_no_trade")
    duplicates += sum(1 for s in normalized_feed_statuses if s == "duplicate_event")
    out_of_order = sum(1 for r in decision_reasons if r == "out_of_order_event_no_trade")
    cas_aux = sum(1 for s in normalized_feed_statuses if s == "cas_auxiliary")
    cas_aux += sum(1 for r in decision_reasons if r == "cas_auxiliary_excluded_no_trade")
    outside = sum(1 for r in decision_reasons if r == "outside_continuous_session_no_trade")

    received_times = [item for item in (_parse_dt(v) for v in normalized_received_ats) if item]
    last_received = max(received_times) if received_times else None
    now = _parse_dt(now_iso) if now_iso else None
    seconds_since: float | None = None
    if last_received is not None and now is not None:
        seconds_since = (now - last_received).total_seconds()

    # Recovery: trailing consecutive ok (non-anomalous) normalized events.
    anomalous = {
        "feed_gap",
        "poll_failure",
        "duplicate_event",
        "malformed_quote",
        "clock_timezone_mismatch",
        "ohlc_violation",
        "invalid_event",
        "missing_instrument",
    }
    trailing_ok = 0
    for status in reversed(normalized_feed_statuses):
        if status in anomalous or status == "cas_auxiliary":
            break
        trailing_ok += 1
    anomalous_decisions = {
        "feed_gap_no_trade",
        "feed_disconnected_no_trade",
        "duplicate_event_no_trade",
        "out_of_order_event_no_trade",
        "cas_auxiliary_excluded_no_trade",
        "outside_continuous_session_no_trade",
        "stale_quote_no_trade",
        "signal_expired_next_session_no_trade",
    }
    trailing_ok_decisions = 0
    for reason in reversed(decision_reasons):
        if reason in anomalous_decisions:
            break
        trailing_ok_decisions += 1
    trailing_ok = min(trailing_ok, trailing_ok_decisions) if decision_reasons else trailing_ok
    has_anomaly = any(
        [
            feed_gaps > 0,
            reconnects > 0,
            duplicates > limits.max_duplicates_tolerated,
            out_of_order > limits.max_out_of_order_tolerated,
            cas_aux > 0,
            outside > 0,
        ]
    )
    recovered = (not has_anomaly) or (trailing_ok >= limits.required_recovery_polls)

    failed_reasons: list[str] = []
    if killed:
        failed_reasons.append("operator_kill_sentinel_present")
    if disk_error:
        failed_reasons.append("disk_write_error_recorded")
    if not persistence_ok:
        failed_reasons.append("persistence_incomplete")
    if checksums_ok is False:
        failed_reasons.append("checksum_mismatch_evidence_corruption")
    if reports_ok is False:
        failed_reasons.append("report_fingerprint_mismatch")

    degraded_reasons: list[str] = []
    if session_closed:
        degraded_reasons.append("session_closed_no_new_trades")
    if last_quote_age_seconds is not None and last_quote_age_seconds > limits.freshness_seconds:
        degraded_reasons.append(
            f"stale_last_quote_age_{last_quote_age_seconds:.1f}s_over_{limits.freshness_seconds:.1f}s"
        )
    if (
        seconds_since is not None
        and expected_cadence_seconds is not None
        and seconds_since > expected_cadence_seconds * limits.cadence_multiplier
    ):
        degraded_reasons.append(
            f"cadence_breach_{seconds_since:.1f}s_since_last_quote_over_"
            f"{expected_cadence_seconds * limits.cadence_multiplier:.1f}s"
        )
    if feed_gaps and not recovered:
        degraded_reasons.append(f"unresolved_feed_gap_count_{feed_gaps}")
    elif feed_gaps:
        reasons.append(f"recovered_feed_gap_count_{feed_gaps}_not_hidden")
    if reconnects and not recovered:
        degraded_reasons.append(f"unresolved_reconnect_boundary_count_{reconnects}")
    elif reconnects:
        reasons.append(f"recovered_reconnect_count_{reconnects}_not_hidden")
    if duplicates > limits.max_duplicates_tolerated and not recovered:
        degraded_reasons.append(f"duplicate_events_{duplicates}_over_tolerance")
    if out_of_order > limits.max_out_of_order_tolerated and not recovered:
        degraded_reasons.append(f"out_of_order_events_{out_of_order}_over_tolerance")
    if cas_aux and not recovered:
        degraded_reasons.append(f"cas_uncertainty_auxiliary_events_{cas_aux}")
    if outside and not recovered:
        degraded_reasons.append(f"post_continuous_events_{outside}")
    if not runner_started:
        degraded_reasons.append("runner_not_started")
    if has_anomaly and recovered:
        reasons.append(f"recovered_after_{trailing_ok}_ok_polls_gaps_retained")

    if failed_reasons:
        status = HealthStatus.FAILED_CLOSED
        reasons = failed_reasons + degraded_reasons + reasons
        allow = False
    elif degraded_reasons:
        status = HealthStatus.DEGRADED_NO_TRADING
        reasons = degraded_reasons + reasons
        allow = False
    else:
        status = HealthStatus.HEALTHY
        reasons.append("healthy_fresh_feed_evidence_complete")
        allow = True

    return ShadowSessionHealthReport(
        schema_version=SCHEMA_VERSION,
        session_id=session_id,
        status=status,
        allow_new_theoretical_trades=allow,
        runner_started=runner_started,
        runner_stopped=runner_stopped,
        killed=killed,
        last_quote_received_at=last_received.isoformat() if last_received else None,
        last_quote_age_seconds=last_quote_age_seconds,
        expected_cadence_seconds=expected_cadence_seconds,
        seconds_since_last_quote=seconds_since,
        feed_gaps=feed_gaps,
        reconnect_boundaries=reconnects,
        duplicates=duplicates,
        out_of_order=out_of_order,
        cas_auxiliary_events=cas_aux,
        outside_continuous_events=outside,
        session_closed=session_closed,
        decisions_count=decisions_count,
        theoretical_trades_count=theoretical_trades_count,
        persistence_ok=persistence_ok,
        checksums_ok=checksums_ok,
        reports_ok=reports_ok,
        disk_error=disk_error,
        reasons=tuple(reasons),
        live_orders_called=False,
    )


def _verify_checksums(output_dir: Path) -> bool | None:
    checksum_file = output_dir / "CHECKSUMS.sha256"
    if not checksum_file.exists():
        return None
    try:
        lines = checksum_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    entries: dict[str, str] = {}
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            return False
        entries[parts[1]] = parts[0]
    for name, expected in entries.items():
        if name == "CHECKSUMS.sha256":
            continue
        target = output_dir / name
        if not target.is_file():
            return False
        try:
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError:
            return False
        if actual != expected:
            return False
    return True


def check_persisted_session(
    output_dir: Path,
    *,
    now_iso: str | None = None,
    thresholds: HealthThresholds | None = None,
) -> ShadowSessionHealthReport:
    """Read-only health check over persisted shadow evidence (no writes)."""
    limits = thresholds or HealthThresholds(freshness_seconds=60.0)
    killed = (output_dir / KILL_SENTINEL).exists()
    disk_error = (output_dir / DISK_ERROR_SENTINEL).exists()
    config_path = output_dir / "config.json"
    summary_path = output_dir / "summary.json"
    runner_started = config_path.exists()
    runner_stopped = summary_path.exists()
    session_id = output_dir.name
    expected_cadence: float | None = None
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            session_id = str(config.get("session_id", session_id))
            expected_cadence = float(config.get("expected_cadence_seconds"))
        except (OSError, ValueError, TypeError):
            expected_cadence = None

    missing = [name for name in REQUIRED_EVIDENCE_FILES if not (output_dir / name).exists()]
    # report-*.json files are required when a summary exists.
    report_files = sorted(output_dir.glob("report-*.json"))
    persistence_ok = not missing and (not runner_stopped or bool(report_files)) and not disk_error

    feed_statuses: list[str] = []
    received_ats: list[str | None] = []
    reconnects: list[bool] = []
    last_age: float | None = None
    events_path = output_dir / "market_events.jsonl"
    if events_path.exists():
        try:
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                feed_statuses.append(str(row.get("feed_status", "unknown")))
                received = row.get("received_timestamp")
                received_ats.append(str(received) if received else None)
                reconnects.append(bool(row.get("reconnect_boundary", False)))
                age = row.get("quote_age_seconds")
                if isinstance(age, (int, float)):
                    last_age = float(age)
        except (OSError, ValueError):
            persistence_ok = False

    decision_reasons: list[str] = []
    decisions_count = 0
    decisions_path = output_dir / "decisions.jsonl"
    if decisions_path.exists():
        try:
            for line in decisions_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                decisions_count += 1
                reason = row.get("reason")
                if reason:
                    decision_reasons.append(str(reason))
        except (OSError, ValueError):
            persistence_ok = False

    trades_count = 0
    trades_path = output_dir / "trades.jsonl"
    if trades_path.exists():
        try:
            trades_count = sum(
                1 for line in trades_path.read_text(encoding="utf-8").splitlines() if line.strip()
            )
        except OSError:
            persistence_ok = False

    checksums_ok = _verify_checksums(output_dir)
    if checksums_ok is None and runner_stopped:
        persistence_ok = False

    reports_ok: bool | None = None
    if report_files and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            expected_fps = summary.get("report_fingerprints", {})
            reports_ok = True
            for path in report_files:
                report = json.loads(path.read_text(encoding="utf-8"))
                payload = {k: v for k, v in report.items() if k != "fingerprint"}
                digest = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                if report.get("fingerprint") != digest:
                    reports_ok = False
                for key, expected in expected_fps.items():
                    safe = key.replace("|", "_").replace(":", "_")
                    if path.name == f"report-{safe}.json" and expected != digest:
                        reports_ok = False
        except (OSError, ValueError, TypeError):
            reports_ok = False
            persistence_ok = False

    session_closed = any(
        status in ("cas_auxiliary",) or reason == "outside_continuous_session_no_trade"
        for status, reason in zip(feed_statuses, decision_reasons, strict=False)
    ) or any(r == "outside_continuous_session_no_trade" for r in decision_reasons)

    now = now_iso or datetime.now(ZoneInfo(RUNNER_TIMEZONE)).isoformat()
    return evaluate_session_health(
        session_id=session_id,
        runner_started=runner_started,
        runner_stopped=runner_stopped,
        killed=killed,
        disk_error=disk_error,
        persistence_ok=persistence_ok,
        checksums_ok=checksums_ok,
        reports_ok=reports_ok,
        normalized_feed_statuses=tuple(feed_statuses),
        normalized_received_ats=tuple(received_ats),
        normalized_reconnects=tuple(reconnects),
        decision_reasons=tuple(decision_reasons),
        decisions_count=decisions_count,
        theoretical_trades_count=trades_count,
        session_closed=session_closed,
        expected_cadence_seconds=expected_cadence,
        last_quote_age_seconds=last_age,
        now_iso=now,
        thresholds=limits,
    )


__all__ = [
    "DISK_ERROR_SENTINEL",
    "KILL_SENTINEL",
    "SCHEMA_VERSION",
    "HealthStatus",
    "HealthThresholds",
    "ShadowSessionHealthReport",
    "check_persisted_session",
    "evaluate_session_health",
]
