"""Shadow evidence qualification V1.

Consumes persisted read-only live shadow-session artifacts and qualifies them
into deterministic paper/shadow evidence suitable for the existing
experiment/promotion evaluation lifecycle.

Safety contract (load-bearing):
- This module contains NO order method. It never calls a broker, never places
  an order, and never mutates real capital.
- Only consumes read-only offline persisted session artifacts.
- UNKNOWN != ZERO: missing data, missing files, or corrupted checksums fail closed.
- A positive P&L does NOT imply qualification. Evidence quality, feed continuity,
  coverage, replay consistency, and checksum verification are strictly independent
  from performance.
- Any attempt to qualify a session with live_orders_called=True fails closed with
  LiveOrderAttemptError.
- PaperTradingEvidence adapter is constructed ONLY when evidence meets or exceeds
  caller-supplied qualification thresholds (SUFFICIENT_FOR_REVIEW) and contains
  at least one verified theoretical order.
- SUFFICIENT_FOR_REVIEW is NOT live authorization.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any

from .experiment import (
    LiveOrderAttemptError,
    PaperTradingEvidence,
    canonical_sha256,
)
from .live_market_readiness import (
    REQUIRED_TIMEZONE,
    LiveMarketReadinessReport,
    ReadinessClassification,
    build_runner_readiness_context,
)
from .shadow_live_runner import (
    FeedMode,
    RunnerMode,
    ShadowLiveConfig,
    verify_persisted_replay,
)
from .shadow_session_health import HealthStatus, HealthThresholds, check_persisted_session

SCHEMA_VERSION = "shadow-evidence-qualification/v1"


class QualificationError(Exception):
    """Base exception for shadow evidence qualification failures."""


class MissingSessionDataError(QualificationError):
    """Raised when required session files or checksums are missing."""


class ChecksumMismatchError(QualificationError):
    """Raised when persisted artifact digests do not match CHECKSUMS.sha256."""


class ReplayMismatchError(QualificationError):
    """Raised when deterministic replay does not match persisted reports."""


class QualificationNotSufficientError(QualificationError):
    """Raised when attempting to adapt unqualified evidence to PaperTradingEvidence."""


class ZeroTradesPaperEvidenceError(QualificationError):
    """Raised when evidence has zero trades, which cannot satisfy PaperTradingEvidence."""


class ShadowEvidenceClassification(StrEnum):
    """Qualification status of evaluated shadow evidence."""

    INVALID_SHADOW_EVIDENCE = "INVALID_SHADOW_EVIDENCE"
    VALID_SHADOW_EVIDENCE_INSUFFICIENT = "VALID_SHADOW_EVIDENCE_INSUFFICIENT"
    VALID_SHADOW_EVIDENCE_SUFFICIENT_FOR_REVIEW = "VALID_SHADOW_EVIDENCE_SUFFICIENT_FOR_REVIEW"


@dataclass(frozen=True)
class ShadowQualificationPolicy:
    """Caller-supplied threshold policy for shadow evidence qualification.

    Strictly no hidden numeric defaults: all required threshold parameters must be
    explicitly provided by the caller.
    """

    policy_id: str
    min_valid_shadow_sessions: int
    min_observed_coverage_ratio: Decimal
    max_feed_gap_count: int
    max_stale_event_count: int
    min_theoretical_trades: int
    max_replay_mismatches: int
    require_checksum_verification: bool
    require_replay_verification: bool
    allow_zero_trades: bool
    require_zero_live_orders: bool = True
    max_drawdown_rupees: Decimal | None = None
    require_cost_reconciliation: bool = False
    max_cost_reconciliation_error_inr: Decimal | None = None

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ValueError("policy_id is required")
        if self.min_valid_shadow_sessions <= 0:
            raise ValueError("min_valid_shadow_sessions must be positive")
        if not (Decimal(0) <= self.min_observed_coverage_ratio <= Decimal(1)):
            raise ValueError("min_observed_coverage_ratio must be in [0, 1]")
        if self.max_feed_gap_count < 0:
            raise ValueError("max_feed_gap_count cannot be negative")
        if self.max_stale_event_count < 0:
            raise ValueError("max_stale_event_count cannot be negative")
        if self.min_theoretical_trades < 0:
            raise ValueError("min_theoretical_trades cannot be negative")
        if self.max_replay_mismatches < 0:
            raise ValueError("max_replay_mismatches cannot be negative")
        if not self.require_zero_live_orders:
            raise ValueError(
                "require_zero_live_orders must be True; live orders strictly forbidden"
            )
        if self.max_drawdown_rupees is not None and self.max_drawdown_rupees < Decimal(0):
            raise ValueError("max_drawdown_rupees cannot be negative")
        if (
            self.max_cost_reconciliation_error_inr is not None
            and self.max_cost_reconciliation_error_inr < Decimal(0)
        ):
            raise ValueError("max_cost_reconciliation_error_inr cannot be negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "min_valid_shadow_sessions": self.min_valid_shadow_sessions,
            "min_observed_coverage_ratio": str(self.min_observed_coverage_ratio),
            "max_feed_gap_count": self.max_feed_gap_count,
            "max_stale_event_count": self.max_stale_event_count,
            "min_theoretical_trades": self.min_theoretical_trades,
            "max_replay_mismatches": self.max_replay_mismatches,
            "require_checksum_verification": self.require_checksum_verification,
            "require_replay_verification": self.require_replay_verification,
            "allow_zero_trades": self.allow_zero_trades,
            "require_zero_live_orders": self.require_zero_live_orders,
            "max_drawdown_rupees": (
                str(self.max_drawdown_rupees) if self.max_drawdown_rupees is not None else None
            ),
            "require_cost_reconciliation": self.require_cost_reconciliation,
            "max_cost_reconciliation_error_inr": (
                str(self.max_cost_reconciliation_error_inr)
                if self.max_cost_reconciliation_error_inr is not None
                else None
            ),
        }

    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())


@dataclass(frozen=True)
class InstrumentCoverageMetrics:
    """Market-event observation coverage per instrument."""

    instrument_key: str
    expected_events: int
    observed_events: int
    valid_events: int
    coverage_ratio: Decimal

    def as_dict(self) -> dict[str, Any]:
        return {
            "instrument_key": self.instrument_key,
            "expected_events": self.expected_events,
            "observed_events": self.observed_events,
            "valid_events": self.valid_events,
            "coverage_ratio": str(self.coverage_ratio),
        }


@dataclass(frozen=True)
class ShadowSessionMetrics:
    """Comprehensive measurement and integrity audit of one persisted shadow session."""

    session_id: str
    session_date: date
    session_fingerprint: str
    instrument_keys: tuple[str, ...]
    instrument_coverages: tuple[InstrumentCoverageMetrics, ...]
    expected_market_events: int
    observed_market_events: int
    valid_market_events: int
    observed_coverage_ratio: Decimal
    data_gaps: int
    stale_events: int
    reconnect_events: int
    duplicate_or_out_of_order_events: int
    cas_exclusions: int
    strategy_identity: str
    cost_scenario_identity: str
    approved_capital_identity: str
    theoretical_trade_count: int
    theoretical_win_loss_count: tuple[int, int]
    gross_pnl: Decimal
    modeled_costs: Decimal
    net_theoretical_pnl: Decimal
    max_drawdown_rupees: Decimal
    no_trade_decisions_by_reason: dict[str, int]
    theoretical_only_confirmation: bool
    live_orders_called: bool
    replay_verification: dict[str, bool]
    checksum_verification: dict[str, bool]
    integrity_violations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.live_orders_called:
            raise LiveOrderAttemptError(
                f"shadow session {self.session_id} reports live_orders_called=True; "
                "live orders are strictly forbidden in research/shadow evaluation"
            )

    @property
    def is_valid(self) -> bool:
        """True only if session has zero integrity violations and all checks passed."""
        return (
            len(self.integrity_violations) == 0
            and not self.live_orders_called
            and self.theoretical_only_confirmation
            and len(self.checksum_verification) > 0
            and all(self.checksum_verification.values())
            and len(self.replay_verification) > 0
            and all(self.replay_verification.values())
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_date": self.session_date.isoformat(),
            "session_fingerprint": self.session_fingerprint,
            "instrument_keys": list(self.instrument_keys),
            "instrument_coverages": [item.as_dict() for item in self.instrument_coverages],
            "expected_market_events": self.expected_market_events,
            "observed_market_events": self.observed_market_events,
            "valid_market_events": self.valid_market_events,
            "observed_coverage_ratio": str(self.observed_coverage_ratio),
            "data_gaps": self.data_gaps,
            "stale_events": self.stale_events,
            "reconnect_events": self.reconnect_events,
            "duplicate_or_out_of_order_events": self.duplicate_or_out_of_order_events,
            "cas_exclusions": self.cas_exclusions,
            "strategy_identity": self.strategy_identity,
            "cost_scenario_identity": self.cost_scenario_identity,
            "approved_capital_identity": self.approved_capital_identity,
            "theoretical_trade_count": self.theoretical_trade_count,
            "theoretical_win_loss_count": list(self.theoretical_win_loss_count),
            "gross_pnl": str(self.gross_pnl),
            "modeled_costs": str(self.modeled_costs),
            "net_theoretical_pnl": str(self.net_theoretical_pnl),
            "max_drawdown_rupees": str(self.max_drawdown_rupees),
            "no_trade_decisions_by_reason": dict(sorted(self.no_trade_decisions_by_reason.items())),
            "theoretical_only_confirmation": self.theoretical_only_confirmation,
            "live_orders_called": False,
            "replay_verification": dict(sorted(self.replay_verification.items())),
            "checksum_verification": dict(sorted(self.checksum_verification.items())),
            "integrity_violations": list(self.integrity_violations),
        }


@dataclass(frozen=True)
class ShadowEvidenceQualification:
    """Formal qualification result across one or more shadow sessions."""

    schema_version: str
    policy: ShadowQualificationPolicy
    session_metrics: tuple[ShadowSessionMetrics, ...]
    classification: ShadowEvidenceClassification
    qualification_passed: bool
    qualification_reasons: tuple[str, ...]
    total_sessions_evaluated: int
    valid_sessions_count: int
    aggregate_coverage_ratio: Decimal
    total_data_gaps: int
    total_stale_events: int
    total_reconnect_events: int
    total_duplicate_or_out_of_order_events: int
    total_cas_exclusions: int
    total_theoretical_trades: int
    total_theoretical_wins: int
    total_theoretical_losses: int
    total_gross_pnl: Decimal
    total_modeled_costs: Decimal
    total_net_theoretical_pnl: Decimal
    max_aggregate_drawdown_rupees: Decimal
    live_orders_called: bool = False
    theoretical_only_confirmation: bool = True

    def __post_init__(self) -> None:
        if self.live_orders_called:
            raise LiveOrderAttemptError(
                "qualification result cannot have live_orders_called=True; "
                "live orders are strictly forbidden"
            )

    def qualification_fingerprint(self) -> str:
        """Deterministic hash of the full qualification result."""
        payload = {
            "schema_version": self.schema_version,
            "policy_fingerprint": self.policy.fingerprint(),
            "session_fingerprints": [m.session_fingerprint for m in self.session_metrics],
            "classification": self.classification.value,
            "qualification_passed": self.qualification_passed,
            "qualification_reasons": list(self.qualification_reasons),
            "total_sessions_evaluated": self.total_sessions_evaluated,
            "valid_sessions_count": self.valid_sessions_count,
            "aggregate_coverage_ratio": str(self.aggregate_coverage_ratio),
            "total_data_gaps": self.total_data_gaps,
            "total_stale_events": self.total_stale_events,
            "total_reconnect_events": self.total_reconnect_events,
            "total_duplicate_or_out_of_order_events": self.total_duplicate_or_out_of_order_events,
            "total_cas_exclusions": self.total_cas_exclusions,
            "total_theoretical_trades": self.total_theoretical_trades,
            "total_theoretical_wins": self.total_theoretical_wins,
            "total_theoretical_losses": self.total_theoretical_losses,
            "total_gross_pnl": str(self.total_gross_pnl),
            "total_modeled_costs": str(self.total_modeled_costs),
            "total_net_theoretical_pnl": str(self.total_net_theoretical_pnl),
            "max_aggregate_drawdown_rupees": str(self.max_aggregate_drawdown_rupees),
            "live_orders_called": False,
            "theoretical_only_confirmation": True,
        }
        return canonical_sha256(payload)

    def to_paper_trading_evidence(self) -> PaperTradingEvidence:
        """Adapt qualified shadow evidence to canonical PaperTradingEvidence.

        Can ONLY be called when classification is VALID_SHADOW_EVIDENCE_SUFFICIENT_FOR_REVIEW
        and at least one verified theoretical order was executed.
        """
        if (
            self.classification
            is not ShadowEvidenceClassification.VALID_SHADOW_EVIDENCE_SUFFICIENT_FOR_REVIEW
        ):
            raise QualificationNotSufficientError(
                f"cannot adapt to PaperTradingEvidence: evidence classification is "
                f"{self.classification.value}; violations: " + "; ".join(self.qualification_reasons)
            )
        if self.total_theoretical_trades <= 0:
            raise ZeroTradesPaperEvidenceError(
                "cannot adapt to PaperTradingEvidence: PaperTradingEvidence requires at least "
                f"one verified trade (verified_orders_count > 0), found {self.total_theoretical_trades}"
            )
        min_date = min(m.session_date for m in self.session_metrics)
        max_date = max(m.session_date for m in self.session_metrics)
        audit_payload = {
            "policy_fingerprint": self.policy.fingerprint(),
            "session_fingerprints": [m.session_fingerprint for m in self.session_metrics],
            "qualification_fingerprint": self.qualification_fingerprint(),
        }
        audit_hash = canonical_sha256(audit_payload)
        return PaperTradingEvidence(
            artifact_fingerprint=self.qualification_fingerprint(),
            environment="shadow-live-read-only",
            session_start=min_date,
            session_end=max_date,
            verified_orders_count=self.total_theoretical_trades,
            audit_log_fingerprint=audit_hash,
            source_reference=f"shadow-qualification:{self.policy.policy_id}",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "qualification_fingerprint": self.qualification_fingerprint(),
            "policy": self.policy.as_dict(),
            "classification": self.classification.value,
            "qualification_passed": self.qualification_passed,
            "qualification_reasons": list(self.qualification_reasons),
            "total_sessions_evaluated": self.total_sessions_evaluated,
            "valid_sessions_count": self.valid_sessions_count,
            "aggregate_coverage_ratio": str(self.aggregate_coverage_ratio),
            "total_data_gaps": self.total_data_gaps,
            "total_stale_events": self.total_stale_events,
            "total_reconnect_events": self.total_reconnect_events,
            "total_duplicate_or_out_of_order_events": self.total_duplicate_or_out_of_order_events,
            "total_cas_exclusions": self.total_cas_exclusions,
            "total_theoretical_trades": self.total_theoretical_trades,
            "total_theoretical_wins": self.total_theoretical_wins,
            "total_theoretical_losses": self.total_theoretical_losses,
            "total_gross_pnl": str(self.total_gross_pnl),
            "total_modeled_costs": str(self.total_modeled_costs),
            "total_net_theoretical_pnl": str(self.total_net_theoretical_pnl),
            "max_aggregate_drawdown_rupees": str(self.max_aggregate_drawdown_rupees),
            "live_orders_called": False,
            "theoretical_only_confirmation": True,
            "sessions": [m.as_dict() for m in self.session_metrics],
        }


def inspect_shadow_session(
    session_dir: Path | str,
    *,
    expected_session_date: date | None = None,
) -> ShadowSessionMetrics:
    """Inspect and measure a persisted shadow session directory.

    Fails closed to invalid status if any file is missing, corrupted, checksum-mismatched,
    replay-mismatched, or if live orders were called.
    """
    output_dir = Path(session_dir)
    violations: list[str] = []
    checksum_results: dict[str, bool] = {}
    replay_results: dict[str, bool] = {}

    if not output_dir.exists() or not output_dir.is_dir():
        violations.append(f"missing_session_directory: {output_dir}")
        return _fallback_invalid_metrics("unknown_missing", violations)

    required_files = (
        "config.json",
        "readiness-report.json",
        "market_events.jsonl",
        "decisions.jsonl",
        "trades.jsonl",
        "summary.json",
        "CHECKSUMS.sha256",
        "health-report.json",
    )
    for fname in required_files:
        if not (output_dir / fname).is_file():
            violations.append(f"missing_required_file: {fname}")

    # Verify CHECKSUMS.sha256
    checksums_path = output_dir / "CHECKSUMS.sha256"
    if checksums_path.is_file():
        checksum_text = checksums_path.read_text(encoding="utf-8").strip()
        if not checksum_text:
            violations.append("empty_checksums_file")
        else:
            for line in checksum_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    violations.append(f"malformed_checksum_line: {line}")
                    continue
                expected_digest, target_name = parts
                target_name = target_name.strip()
                target_file = output_dir / target_name
                if not target_file.is_file():
                    checksum_results[target_name] = False
                    violations.append(f"checksum_target_missing: {target_name}")
                else:
                    actual_digest = hashlib.sha256(target_file.read_bytes()).hexdigest()
                    if actual_digest != expected_digest:
                        checksum_results[target_name] = False
                        violations.append(f"checksum_mismatch: {target_name}")
                    else:
                        checksum_results[target_name] = True

            persisted_targets = {
                path.name
                for path in output_dir.iterdir()
                if path.is_file() and path.name != "CHECKSUMS.sha256"
            }
            for target_name in sorted(persisted_targets - checksum_results.keys()):
                checksum_results[target_name] = False
                violations.append(f"checksum_manifest_missing_target: {target_name}")

    # Parse config.json
    config: ShadowLiveConfig | None = None
    config_path = output_dir / "config.json"
    session_id = "unknown"
    instrument_keys: tuple[str, ...] = ()
    if config_path.is_file():
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
            if config_data.get("live_orders_called") is not False:
                raise LiveOrderAttemptError(
                    "config.json contains live_orders_called=True; live orders are strictly forbidden"
                )
            session_id = str(config_data.get("session_id", "unknown"))
            instrument_keys = tuple(config_data.get("instrument_keys", ()))
            cas_tuples = tuple(
                (item["instrument_key"], bool(item["cas_eligible"]))
                for item in config_data["cas_eligible_by_key"]
            )
            tick_tuples = tuple(
                (item["instrument_key"], str(item["tick_size_rupees"]))
                for item in config_data["tick_size_by_key"]
            )
            config = ShadowLiveConfig(
                session_id=session_id,
                instrument_keys=instrument_keys,
                cas_eligible_by_key=cas_tuples,
                tick_size_by_key=tick_tuples,
                feed_mode=FeedMode(config_data["feed_mode"]),
                poll_interval_seconds=float(config_data["poll_interval_seconds"]),
                max_polls=int(config_data["max_polls"]),
                quote_freshness_threshold_seconds=float(
                    config_data["quote_freshness_threshold_seconds"]
                ),
                expected_cadence_seconds=float(config_data["expected_cadence_seconds"]),
                approved_capital_rupees=str(config_data["approved_capital_rupees"]),
                exit_buffer_minutes=int(config_data["exit_buffer_minutes"]),
                strategy_name=str(config_data["strategy_name"]),
                output_dir=str(config_data["output_dir"]),
                mode=RunnerMode(config_data["mode"]),
            )
        except LiveOrderAttemptError:
            raise
        except (json.JSONDecodeError, OSError, KeyError, ValueError, TypeError) as exc:
            violations.append(f"config_parse_error: {exc}")

    # Check report files per instrument
    for key in instrument_keys:
        safe_key = key.replace("|", "_").replace(":", "_")
        report_file = output_dir / f"report-{safe_key}.json"
        if not report_file.is_file():
            violations.append(f"missing_report_file: report-{safe_key}.json")

    readiness_report: LiveMarketReadinessReport | None = None
    readiness_path = output_dir / "readiness-report.json"
    if readiness_path.is_file():
        try:
            readiness_payload = json.loads(readiness_path.read_text(encoding="utf-8"))
            if not isinstance(readiness_payload, dict):
                raise TypeError("readiness report must be an object")
            readiness_report = LiveMarketReadinessReport.from_dict(readiness_payload)
        except (json.JSONDecodeError, OSError, KeyError, ValueError, TypeError) as exc:
            violations.append(f"readiness_report_parse_error: {exc}")

    # Parse summary.json
    summary_data: dict[str, Any] | None = None
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        try:
            loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(loaded_summary, dict):
                raise TypeError("summary must be an object")
            summary_data = loaded_summary
            if summary_data.get("live_orders_called") is not False:
                raise LiveOrderAttemptError(
                    "summary.json contains live_orders_called=True; live orders are strictly forbidden"
                )
        except LiveOrderAttemptError:
            raise
        except (json.JSONDecodeError, OSError, KeyError, ValueError, TypeError) as exc:
            violations.append(f"summary_parse_error: {exc}")

    if readiness_report is not None and summary_data is not None:
        if (
            summary_data.get("readiness_context_fingerprint")
            != readiness_report.context.fingerprint()
        ):
            violations.append("readiness_context_fingerprint_mismatch")
        if summary_data.get("readiness_classification") != readiness_report.classification.value:
            violations.append("readiness_classification_mismatch")
    if readiness_report is not None and readiness_report.classification not in (
        ReadinessClassification.READY_FOR_RESEARCH_SHADOW,
        ReadinessClassification.READY_FOR_LIVE_ORDER_REVIEW,
    ):
        violations.append("readiness_not_research_shadow_capable")

    # The health report is the persisted output of the canonical Monday health
    # monitor. Validate its own fingerprint and compare it with a fresh,
    # read-only recomputation over the same persisted evidence.
    health_payload: dict[str, Any] | None = None
    stored_health_fingerprint: str | None = None
    health_path = output_dir / "health-report.json"
    if health_path.is_file():
        try:
            loaded_health = json.loads(health_path.read_text(encoding="utf-8"))
            if not isinstance(loaded_health, dict):
                raise TypeError("health report must be an object")
            stored_health_fingerprint = loaded_health.pop("fingerprint", None)
            expected_health_fingerprint = hashlib.sha256(
                json.dumps(loaded_health, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if stored_health_fingerprint != expected_health_fingerprint:
                violations.append("health_report_fingerprint_mismatch")
            if loaded_health.get("live_orders_called") is not False:
                raise LiveOrderAttemptError(
                    "health-report.json contains live_orders_called=True; "
                    "live orders are strictly forbidden"
                )
            if loaded_health.get("status") not in {item.value for item in HealthStatus}:
                violations.append("health_report_status_invalid")
            health_payload = loaded_health
            if summary_data is not None and summary_data.get("health_status") != loaded_health.get(
                "status"
            ):
                violations.append("health_status_mismatch")
            if (
                summary_data is not None
                and summary_data.get("health_report_fingerprint") != stored_health_fingerprint
            ):
                violations.append("health_report_summary_fingerprint_mismatch")
        except LiveOrderAttemptError:
            raise
        except (json.JSONDecodeError, OSError, KeyError, ValueError, TypeError) as exc:
            violations.append(f"health_report_parse_error: {exc}")

    # Parse decisions.jsonl
    no_trade_decisions: dict[str, int] = {}
    decisions_path = output_dir / "decisions.jsonl"
    if decisions_path.is_file():
        try:
            for line in decisions_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("live_orders_called") is not False:
                    raise LiveOrderAttemptError(
                        "decisions.jsonl contains live_orders_called=True; "
                        "live orders are strictly forbidden"
                    )
                reason = str(row.get("reason", "unknown"))
                no_trade_decisions[reason] = no_trade_decisions.get(reason, 0) + 1
        except LiveOrderAttemptError:
            raise
        except (json.JSONDecodeError, OSError, KeyError, ValueError, TypeError) as exc:
            violations.append(f"decisions_parse_error: {exc}")

    # Parse trades.jsonl
    trades_list: list[dict[str, Any]] = []
    theoretical_only = True
    trades_path = output_dir / "trades.jsonl"
    if trades_path.is_file():
        try:
            for line in trades_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                label = row.get("label")
                if label not in ("theoretical_never_broker_confirmed", "theoretical"):
                    theoretical_only = False
                    violations.append(f"trade_not_theoretical_only: {label}")
                trades_list.append(row)
        except (json.JSONDecodeError, OSError, KeyError, ValueError, TypeError) as exc:
            violations.append(f"trades_parse_error: {exc}")

    # Parse reports and extract report metrics
    reports_by_key: dict[str, dict[str, Any]] = {}
    strategy_identity = "unknown"
    cost_scenario_identity = "unknown"
    approved_capital_identity = "unknown"
    max_drawdown = Decimal(0)
    identity_values: dict[str, set[str]] = {
        "strategy_identity": set(),
        "cost_scenario_identity": set(),
        "approved_capital_identity": set(),
    }
    for key in instrument_keys:
        safe_key = key.replace("|", "_").replace(":", "_")
        report_file = output_dir / f"report-{safe_key}.json"
        if report_file.is_file():
            try:
                rep = json.loads(report_file.read_text(encoding="utf-8"))
                if rep.get("live_orders_called") is not False:
                    raise LiveOrderAttemptError(
                        f"report-{safe_key}.json contains live_orders_called=True; "
                        "live orders are strictly forbidden"
                    )
                strategy_identity = str(rep.get("strategy_identity", strategy_identity))
                cost_scenario_identity = str(
                    rep.get("cost_scenario_fingerprint", cost_scenario_identity)
                )
                approved_capital_identity = str(
                    rep.get("approved_capital_identity", approved_capital_identity)
                )
                for identity_key, identity_value in (
                    ("strategy_identity", strategy_identity),
                    ("cost_scenario_identity", cost_scenario_identity),
                    ("approved_capital_identity", approved_capital_identity),
                ):
                    if identity_value.strip() and identity_value != "unknown":
                        identity_values[identity_key].add(identity_value)
                if "max_drawdown_rupees" not in rep:
                    violations.append(f"report_missing_drawdown_evidence: {safe_key}")
                    dd = Decimal(0)
                else:
                    dd = Decimal(str(rep["max_drawdown_rupees"]))
                max_drawdown = max(max_drawdown, dd)
                reports_by_key[key] = rep
            except LiveOrderAttemptError:
                raise
            except (json.JSONDecodeError, OSError, KeyError, ValueError, TypeError) as exc:
                violations.append(f"report_parse_error_{safe_key}: {exc}")

    # Parse market_events.jsonl
    events_by_key: dict[str, list[dict[str, Any]]] = {k: [] for k in instrument_keys}
    data_gaps = 0
    stale_events = 0
    reconnect_events = 0
    duplicate_or_out_of_order_events = 0
    cas_exclusions = 0
    session_date = expected_session_date or date(2026, 9, 7)
    latest_received: datetime | None = None

    events_path = output_dir / "market_events.jsonl"
    if events_path.is_file():
        try:
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row.get("instrument_key", ""))
                if key in events_by_key:
                    events_by_key[key].append(row)
                status = str(row.get("feed_status", ""))
                reason = str(row.get("reason", ""))
                if status == "feed_gap":
                    data_gaps += 1
                if status == "stale_quote":
                    stale_events += 1
                if bool(row.get("reconnect_boundary")) or status == "poll_failure":
                    reconnect_events += 1
                if status in ("duplicate_event", "out_of_order_event"):
                    duplicate_or_out_of_order_events += 1
                if status == "cas_auxiliary" or (
                    row.get("event") and row["event"].get("is_cas_auxiliary")
                ):
                    cas_exclusions += 1
                received_timestamp = row.get("received_timestamp")
                if received_timestamp:
                    try:
                        received = datetime.fromisoformat(str(received_timestamp))
                        if received.tzinfo is not None:
                            latest_received = max(latest_received or received, received)
                    except ValueError:
                        violations.append("market_event_received_timestamp_invalid")
                # Infer session date from source timestamp if not passed
                if expected_session_date is None and row.get("source_timestamp"):
                    try:
                        session_date = datetime.fromisoformat(row["source_timestamp"]).date()
                    except (ValueError, TypeError, KeyError):
                        session_date = expected_session_date or date(2026, 9, 7)
        except (json.JSONDecodeError, OSError, KeyError, ValueError, TypeError) as exc:
            violations.append(f"market_events_parse_error: {exc}")

    if readiness_report is not None and config is not None:
        try:
            expected_context = build_runner_readiness_context(
                trade_date=session_date,
                timezone_name=REQUIRED_TIMEZONE,
                instrument_keys=config.instrument_keys,
                cas_eligible_by_key=config.cas_eligible_by_key,
                tick_size_by_key=config.tick_size_by_key,
                exit_buffer_minutes=config.exit_buffer_minutes,
                approved_capital=config.approved_capital(),
                quote_freshness_threshold_seconds=config.quote_freshness_threshold_seconds,
            )
            if readiness_report.trade_date != session_date:
                violations.append("readiness_trade_date_mismatch")
            if readiness_report.context != expected_context:
                violations.append("readiness_context_mismatch")
            if readiness_report.approved_capital_rupees != config.approved_capital().amount_rupees:
                violations.append("readiness_capital_mismatch")
            checked_at = readiness_report.checked_at_ist
            if checked_at.tzinfo is None:
                violations.append("readiness_checked_at_timezone_missing")
            elif latest_received is not None:
                readiness_age = (
                    latest_received.astimezone(checked_at.tzinfo) - checked_at
                ).total_seconds()
                if (
                    readiness_age < 0
                    or readiness_age > readiness_report.context.readiness_max_age_seconds
                ):
                    violations.append("readiness_stale")
        except (KeyError, TypeError, ValueError) as exc:
            violations.append(f"readiness_context_verification_exception: {exc}")

    # Incorporate engine-level decision reasons into anomaly counts
    data_gaps += no_trade_decisions.get("feed_gap_no_trade", 0)
    stale_events += no_trade_decisions.get("stale_quote_no_trade", 0)
    reconnect_events += no_trade_decisions.get("feed_disconnected_no_trade", 0)
    duplicate_or_out_of_order_events += no_trade_decisions.get(
        "duplicate_event_no_trade", 0
    ) + no_trade_decisions.get("out_of_order_event_no_trade", 0)
    cas_exclusions += no_trade_decisions.get("cas_auxiliary_excluded_no_trade", 0)

    for identity_key, values in identity_values.items():
        if not values:
            violations.append(f"missing_{identity_key}")
        elif len(values) > 1:
            violations.append(f"mismatched_{identity_key}")

    if health_payload is not None:
        try:
            freshness_seconds = (
                float(config.quote_freshness_threshold_seconds) if config is not None else 60.0
            )
            recomputed_health = check_persisted_session(
                output_dir,
                now_iso=(latest_received.isoformat() if latest_received is not None else None),
                thresholds=HealthThresholds(freshness_seconds=freshness_seconds),
            )
            if health_payload != recomputed_health.as_dict():
                violations.append("health_report_mismatch")
            if stored_health_fingerprint != recomputed_health.fingerprint():
                violations.append("health_report_fingerprint_recomputed_mismatch")
            if recomputed_health.status is HealthStatus.FAILED_CLOSED:
                violations.append("health_failed_closed")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            violations.append(f"health_verification_exception: {exc}")

    # Compute coverage per instrument
    max_polls = config.max_polls if config is not None else 1
    total_expected = max_polls * len(instrument_keys)
    total_observed = sum(len(evs) for evs in events_by_key.values())
    total_valid = 0
    instrument_coverages: list[InstrumentCoverageMetrics] = []

    for key in instrument_keys:
        evs = events_by_key.get(key, [])
        observed = len(evs)
        valid = sum(
            1 for item in evs if item.get("event") is not None and item.get("feed_status") == "ok"
        )
        total_valid += valid
        ratio = Decimal(valid) / Decimal(max_polls) if max_polls > 0 else Decimal(0)
        instrument_coverages.append(
            InstrumentCoverageMetrics(
                instrument_key=key,
                expected_events=max_polls,
                observed_events=observed,
                valid_events=valid,
                coverage_ratio=ratio,
            )
        )

    aggregate_coverage_ratio = (
        Decimal(total_valid) / Decimal(total_expected) if total_expected > 0 else Decimal(0)
    )

    # Perform replay verification
    if config is not None and events_path.is_file():
        try:
            replay_res = verify_persisted_replay(
                output_dir, config=config, session_day=session_date
            )
            for key in instrument_keys:
                matched = bool(replay_res.get("matched", {}).get(key, False))
                replay_results[key] = matched
                if not matched:
                    violations.append(f"replay_mismatch: {key}")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            violations.append(f"replay_verification_exception: {exc}")

    # Calculate trade metrics
    total_gross_pnl = Decimal(0)
    total_net_theoretical_pnl = Decimal(0)
    wins = 0
    losses = 0
    for t in trades_list:
        try:
            required_trade_fields = (
                "theoretical_entry",
                "theoretical_exit",
                "quantity",
                "net_theoretical_pnl",
            )
            missing_trade_fields = [field for field in required_trade_fields if field not in t]
            if missing_trade_fields:
                violations.append("trade_missing_pnl_evidence: " + ",".join(missing_trade_fields))
                continue
            entry = Decimal(str(t["theoretical_entry"]))
            exit_p = Decimal(str(t["theoretical_exit"]))
            qty = int(t["quantity"])
            net_pnl = Decimal(str(t["net_theoretical_pnl"]))
            gross = (exit_p - entry) * qty
            total_gross_pnl += gross
            total_net_theoretical_pnl += net_pnl
            if net_pnl > 0:
                wins += 1
            elif net_pnl < 0:
                losses += 1
        except (InvalidOperation, ValueError) as exc:
            violations.append(f"trade_numeric_parse_error: {exc}")

    total_modeled_costs = total_gross_pnl - total_net_theoretical_pnl

    # Construct deterministic session fingerprint
    file_digests: dict[str, str] = {}
    for p in sorted(output_dir.iterdir()):
        if p.is_file() and p.name != "CHECKSUMS.sha256":
            file_digests[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()

    session_fingerprint_payload = {
        "session_id": session_id,
        "session_date": session_date.isoformat(),
        "instrument_keys": list(instrument_keys),
        "file_digests": file_digests,
        "violations": list(violations),
        "live_orders_called": False,
        "theoretical_only": theoretical_only,
    }
    session_fingerprint = canonical_sha256(session_fingerprint_payload)

    return ShadowSessionMetrics(
        session_id=session_id,
        session_date=session_date,
        session_fingerprint=session_fingerprint,
        instrument_keys=instrument_keys,
        instrument_coverages=tuple(instrument_coverages),
        expected_market_events=total_expected,
        observed_market_events=total_observed,
        valid_market_events=total_valid,
        observed_coverage_ratio=aggregate_coverage_ratio,
        data_gaps=data_gaps,
        stale_events=stale_events,
        reconnect_events=reconnect_events,
        duplicate_or_out_of_order_events=duplicate_or_out_of_order_events,
        cas_exclusions=cas_exclusions,
        strategy_identity=strategy_identity,
        cost_scenario_identity=cost_scenario_identity,
        approved_capital_identity=approved_capital_identity,
        theoretical_trade_count=len(trades_list),
        theoretical_win_loss_count=(wins, losses),
        gross_pnl=total_gross_pnl,
        modeled_costs=total_modeled_costs,
        net_theoretical_pnl=total_net_theoretical_pnl,
        max_drawdown_rupees=max_drawdown,
        no_trade_decisions_by_reason=no_trade_decisions,
        theoretical_only_confirmation=theoretical_only,
        live_orders_called=False,
        replay_verification=replay_results,
        checksum_verification=checksum_results,
        integrity_violations=tuple(violations),
    )


def _fallback_invalid_metrics(session_id: str, violations: list[str]) -> ShadowSessionMetrics:
    return ShadowSessionMetrics(
        session_id=session_id,
        session_date=date(2026, 9, 7),
        session_fingerprint=canonical_sha256({"session_id": session_id, "violations": violations}),
        instrument_keys=(),
        instrument_coverages=(),
        expected_market_events=0,
        observed_market_events=0,
        valid_market_events=0,
        observed_coverage_ratio=Decimal(0),
        data_gaps=0,
        stale_events=0,
        reconnect_events=0,
        duplicate_or_out_of_order_events=0,
        cas_exclusions=0,
        strategy_identity="unknown",
        cost_scenario_identity="unknown",
        approved_capital_identity="unknown",
        theoretical_trade_count=0,
        theoretical_win_loss_count=(0, 0),
        gross_pnl=Decimal(0),
        modeled_costs=Decimal(0),
        net_theoretical_pnl=Decimal(0),
        max_drawdown_rupees=Decimal(0),
        no_trade_decisions_by_reason={},
        theoretical_only_confirmation=False,
        live_orders_called=False,
        replay_verification={},
        checksum_verification={},
        integrity_violations=tuple(violations),
    )


def qualify_shadow_evidence(
    sessions: Sequence[Path | str | ShadowSessionMetrics],
    policy: ShadowQualificationPolicy,
) -> ShadowEvidenceQualification:
    """Evaluate and qualify one or more shadow sessions against a qualification policy.

    Produces:
    - INVALID_SHADOW_EVIDENCE: integrity corruption, checksum mismatch, replay mismatch,
      missing session files, or live orders.
    - VALID_SHADOW_EVIDENCE_INSUFFICIENT: structurally valid and verified, but fails one
      or more caller policy thresholds (coverage, sessions, gaps, stale events, trades).
    - VALID_SHADOW_EVIDENCE_SUFFICIENT_FOR_REVIEW: valid, verified, and meets all policy
      thresholds.
    """
    if policy is None:
        raise ValueError("qualification policy is required")

    evaluated_metrics: list[ShadowSessionMetrics] = []
    for item in sessions:
        if isinstance(item, ShadowSessionMetrics):
            evaluated_metrics.append(item)
        else:
            evaluated_metrics.append(inspect_shadow_session(item))

    if not evaluated_metrics:
        # Empty sessions fail closed
        return ShadowEvidenceQualification(
            schema_version=SCHEMA_VERSION,
            policy=policy,
            session_metrics=(),
            classification=ShadowEvidenceClassification.INVALID_SHADOW_EVIDENCE,
            qualification_passed=False,
            qualification_reasons=("no_sessions_provided",),
            total_sessions_evaluated=0,
            valid_sessions_count=0,
            aggregate_coverage_ratio=Decimal(0),
            total_data_gaps=0,
            total_stale_events=0,
            total_reconnect_events=0,
            total_duplicate_or_out_of_order_events=0,
            total_cas_exclusions=0,
            total_theoretical_trades=0,
            total_theoretical_wins=0,
            total_theoretical_losses=0,
            total_gross_pnl=Decimal(0),
            total_modeled_costs=Decimal(0),
            total_net_theoretical_pnl=Decimal(0),
            max_aggregate_drawdown_rupees=Decimal(0),
            live_orders_called=False,
            theoretical_only_confirmation=True,
        )

    # 1. Check for integrity failures
    integrity_reasons: list[str] = []
    valid_sessions: list[ShadowSessionMetrics] = []

    for m in evaluated_metrics:
        if m.integrity_violations:
            for v in m.integrity_violations:
                integrity_reasons.append(f"session_{m.session_id}:{v}")
        if not m.theoretical_only_confirmation:
            integrity_reasons.append(f"session_{m.session_id}:trade_not_confirmed_theoretical")
        if policy.require_checksum_verification and (
            not m.checksum_verification or not all(m.checksum_verification.values())
        ):
            failed_checks = [k for k, ok in m.checksum_verification.items() if not ok]
            integrity_reasons.append(
                f"session_{m.session_id}:checksum_verification_failed:{failed_checks}"
            )
        if policy.require_replay_verification and (
            not m.replay_verification or not all(m.replay_verification.values())
        ):
            failed_replays = [k for k, ok in m.replay_verification.items() if not ok]
            integrity_reasons.append(
                f"session_{m.session_id}:replay_verification_failed:{failed_replays}"
            )
        if m.is_valid:
            valid_sessions.append(m)

    # Aggregate metrics
    total_expected = sum(m.expected_market_events for m in evaluated_metrics)
    total_valid_events = sum(m.valid_market_events for m in evaluated_metrics)
    agg_coverage = (
        Decimal(total_valid_events) / Decimal(total_expected) if total_expected > 0 else Decimal(0)
    )
    total_gaps = sum(m.data_gaps for m in evaluated_metrics)
    total_stale = sum(m.stale_events for m in evaluated_metrics)
    total_reconnects = sum(m.reconnect_events for m in evaluated_metrics)
    total_dups = sum(m.duplicate_or_out_of_order_events for m in evaluated_metrics)
    total_cas = sum(m.cas_exclusions for m in evaluated_metrics)
    total_trades = sum(m.theoretical_trade_count for m in evaluated_metrics)
    total_wins = sum(m.theoretical_win_loss_count[0] for m in evaluated_metrics)
    total_losses = sum(m.theoretical_win_loss_count[1] for m in evaluated_metrics)
    total_gross = sum(m.gross_pnl for m in evaluated_metrics)
    total_costs = sum(m.modeled_costs for m in evaluated_metrics)
    total_net = sum(m.net_theoretical_pnl for m in evaluated_metrics)
    max_dd = max((m.max_drawdown_rupees for m in evaluated_metrics), default=Decimal(0))

    # If any integrity reasons exist, classification is INVALID
    if integrity_reasons:
        return ShadowEvidenceQualification(
            schema_version=SCHEMA_VERSION,
            policy=policy,
            session_metrics=tuple(evaluated_metrics),
            classification=ShadowEvidenceClassification.INVALID_SHADOW_EVIDENCE,
            qualification_passed=False,
            qualification_reasons=tuple(integrity_reasons),
            total_sessions_evaluated=len(evaluated_metrics),
            valid_sessions_count=len(valid_sessions),
            aggregate_coverage_ratio=agg_coverage,
            total_data_gaps=total_gaps,
            total_stale_events=total_stale,
            total_reconnect_events=total_reconnects,
            total_duplicate_or_out_of_order_events=total_dups,
            total_cas_exclusions=total_cas,
            total_theoretical_trades=total_trades,
            total_theoretical_wins=total_wins,
            total_theoretical_losses=total_losses,
            total_gross_pnl=total_gross,
            total_modeled_costs=total_costs,
            total_net_theoretical_pnl=total_net,
            max_aggregate_drawdown_rupees=max_dd,
            live_orders_called=False,
            theoretical_only_confirmation=True,
        )

    # 2. Check qualification thresholds against policy
    policy_violations: list[str] = []

    if len(valid_sessions) < policy.min_valid_shadow_sessions:
        policy_violations.append(
            f"insufficient valid sessions: {len(valid_sessions)} < required {policy.min_valid_shadow_sessions}"
        )
    if agg_coverage < policy.min_observed_coverage_ratio:
        policy_violations.append(
            f"insufficient observed event coverage: {agg_coverage:.4f} < required {policy.min_observed_coverage_ratio}"
        )
    if total_gaps > policy.max_feed_gap_count:
        policy_violations.append(
            f"excessive data gaps: {total_gaps} > allowed {policy.max_feed_gap_count}"
        )
    if total_stale > policy.max_stale_event_count:
        policy_violations.append(
            f"excessive stale events: {total_stale} > allowed {policy.max_stale_event_count}"
        )
    if total_trades < policy.min_theoretical_trades:
        policy_violations.append(
            f"theoretical trade count {total_trades} < required {policy.min_theoretical_trades}"
        )
    if total_trades == 0 and not policy.allow_zero_trades:
        policy_violations.append("zero theoretical trades not allowed by qualification policy")
    if policy.max_drawdown_rupees is not None and max_dd > policy.max_drawdown_rupees:
        policy_violations.append(
            f"max drawdown ₹{max_dd} exceeds allowed ₹{policy.max_drawdown_rupees}"
        )

    if policy_violations:
        classification = ShadowEvidenceClassification.VALID_SHADOW_EVIDENCE_INSUFFICIENT
        passed = False
        reasons = tuple(policy_violations)
    else:
        classification = ShadowEvidenceClassification.VALID_SHADOW_EVIDENCE_SUFFICIENT_FOR_REVIEW
        passed = True
        reasons = ()

    return ShadowEvidenceQualification(
        schema_version=SCHEMA_VERSION,
        policy=policy,
        session_metrics=tuple(evaluated_metrics),
        classification=classification,
        qualification_passed=passed,
        qualification_reasons=reasons,
        total_sessions_evaluated=len(evaluated_metrics),
        valid_sessions_count=len(valid_sessions),
        aggregate_coverage_ratio=agg_coverage,
        total_data_gaps=total_gaps,
        total_stale_events=total_stale,
        total_reconnect_events=total_reconnects,
        total_duplicate_or_out_of_order_events=total_dups,
        total_cas_exclusions=total_cas,
        total_theoretical_trades=total_trades,
        total_theoretical_wins=total_wins,
        total_theoretical_losses=total_losses,
        total_gross_pnl=total_gross,
        total_modeled_costs=total_costs,
        total_net_theoretical_pnl=total_net,
        max_aggregate_drawdown_rupees=max_dd,
        live_orders_called=False,
        theoretical_only_confirmation=True,
    )


def qualify_single_shadow_session(
    session_dir: Path | str,
    policy: ShadowQualificationPolicy,
) -> ShadowEvidenceQualification:
    """Convenience helper to qualify a single shadow session directory."""
    return qualify_shadow_evidence([session_dir], policy)


__all__ = [
    "SCHEMA_VERSION",
    "ChecksumMismatchError",
    "InstrumentCoverageMetrics",
    "MissingSessionDataError",
    "QualificationError",
    "QualificationNotSufficientError",
    "ReplayMismatchError",
    "ShadowEvidenceClassification",
    "ShadowEvidenceQualification",
    "ShadowQualificationPolicy",
    "ShadowSessionMetrics",
    "ZeroTradesPaperEvidenceError",
    "inspect_shadow_session",
    "qualify_shadow_evidence",
    "qualify_single_shadow_session",
]
