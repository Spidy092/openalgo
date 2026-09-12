"""Tests for Shadow Evidence Qualification V1.

Verifies:
- Checksum corruption fails closed (INVALID_SHADOW_EVIDENCE)
- Replay mismatch fails closed (INVALID_SHADOW_EVIDENCE)
- Insufficient sessions fails closed (VALID_SHADOW_EVIDENCE_INSUFFICIENT)
- Insufficient coverage fails closed (VALID_SHADOW_EVIDENCE_INSUFFICIENT)
- Excessive gap / stale events fail closed (VALID_SHADOW_EVIDENCE_INSUFFICIENT)
- Valid but losing session passes qualification (profitability != qualification)
- Valid profitable session passes qualification, but fails if corrupted
- Zero-trade session behavior (allowed by policy, but blocks PaperTradingEvidence)
- CAS auxiliary exclusions accurately measured and excluded
- Policy mutation strictly changes policy and qualification fingerprints
- Session mutation strictly changes session and qualification fingerprints
- live_orders_called=True is strictly rejected with LiveOrderAttemptError
- Adapter to PaperTradingEvidence works only when SUFFICIENT_FOR_REVIEW and trades > 0
- Qualified shadow session does not automatically satisfy experiment promotion
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from equity_engine.cost_ledger import EffectiveDatedCostLedger, LedgerProduct
from equity_engine.experiment import (
    ApprovedCapital,
    BaselineComparisonEvidence,
    ConcretePromotionEvidence,
    CorporateActionEvidenceIdentity,
    CostEvidenceIdentity,
    CostModelIdentity,
    CostReconciliationEvidence,
    EmbargoSpec,
    EventDrivenSimulationEvidence,
    ExperimentOrchestrator,
    FrictionScenarioSpec,
    HeldOutTestEvidence,
    LiveOrderAttemptError,
    NSEMembershipEvidenceIdentity,
    PaperTradingEvidence,
    ResearchWindowConfig,
    SessionPolicyIdentity,
    SlippageStressEvidence,
    StrategySpec,
    TickEvidenceIdentity,
    WindowSpec,
)
from equity_engine.gates import DrawdownBasis, PromotionThresholds
from equity_engine.live_market_readiness import build_synthetic_readiness_report
from equity_engine.shadow_evidence_qualification import (
    ShadowEvidenceClassification,
    ShadowQualificationPolicy,
    ZeroTradesPaperEvidenceError,
    inspect_shadow_session,
    qualify_single_shadow_session,
)
from equity_engine.shadow_live_runner import (
    FeedMode,
    RunnerMode,
    ShadowLiveConfig,
    ShadowLiveRunner,
    SyntheticQuoteSource,
)
from equity_engine.shadow_session_health import KILL_SENTINEL, HealthStatus

IST = ZoneInfo("Asia/Kolkata")
KEY_A = "NSE_EQ|INE002A01018"
KEY_B = "NSE_EQ|INE009A01021"


def _make_policy(**overrides) -> ShadowQualificationPolicy:
    params: dict[str, object] = {
        "policy_id": "test-qualification-policy-v1",
        "min_valid_shadow_sessions": 1,
        "min_observed_coverage_ratio": Decimal("0.90"),
        "max_feed_gap_count": 0,
        "max_stale_event_count": 0,
        "min_theoretical_trades": 1,
        "max_replay_mismatches": 0,
        "require_checksum_verification": True,
        "require_replay_verification": True,
        "allow_zero_trades": False,
        "require_zero_live_orders": True,
        "max_drawdown_rupees": Decimal(5000),
        "require_cost_reconciliation": False,
        "max_cost_reconciliation_error_inr": None,
    }
    params.update(overrides)
    return ShadowQualificationPolicy(**params)  # type: ignore[arg-type]


def _quote(token: str, ts: str, price: float) -> dict[str, object]:
    return {
        "instrument_token": token,
        "timestamp": ts,
        "last_price": price + 0.1,
        "prev_close_price": 100,
        "ohlc": {
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price + 0.1,
        },
    }


def _run_and_persist_session(
    out_dir: Path,
    *,
    session_id: str = "test-session-1",
    instrument_keys: tuple[str, ...] = (KEY_A,),
    strategy_name: str = "exit-second-bar",
    batches: list[dict[str, dict[str, object]]] | None = None,
    poll_times: list[datetime] | None = None,
    cas_eligible: bool = False,
    max_polls: int | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if batches is None:
        # Default 5 bars: entry signal, entry fill, hold, exit signal, exit fill
        batches = [
            {
                KEY_A: _quote(KEY_A, "2026-09-07T09:15:00+05:30", 100.0),
            },
            {
                KEY_A: _quote(KEY_A, "2026-09-07T09:20:00+05:30", 102.0),
            },
            {
                KEY_A: _quote(KEY_A, "2026-09-07T09:25:00+05:30", 105.0),
            },
            {
                KEY_A: _quote(KEY_A, "2026-09-07T09:30:00+05:30", 108.0),
            },
            {
                KEY_A: _quote(KEY_A, "2026-09-07T09:35:00+05:30", 110.0),
            },
        ]
    if poll_times is None:
        poll_times = [
            datetime.fromisoformat("2026-09-07T09:15:05+05:30"),
            datetime.fromisoformat("2026-09-07T09:20:05+05:30"),
            datetime.fromisoformat("2026-09-07T09:25:05+05:30"),
            datetime.fromisoformat("2026-09-07T09:30:05+05:30"),
            datetime.fromisoformat("2026-09-07T09:35:05+05:30"),
        ]
    times_iter = list(poll_times)

    cfg = ShadowLiveConfig(
        session_id=session_id,
        instrument_keys=instrument_keys,
        cas_eligible_by_key=tuple((k, cas_eligible) for k in instrument_keys),
        tick_size_by_key=tuple((k, "0.05") for k in instrument_keys),
        feed_mode=FeedMode.POLL,
        poll_interval_seconds=0.0,
        max_polls=max_polls or len(batches),
        quote_freshness_threshold_seconds=60.0,
        expected_cadence_seconds=300.0,
        approved_capital_rupees="100000",
        exit_buffer_minutes=15,
        strategy_name=strategy_name,
        output_dir=str(out_dir),
        mode=RunnerMode.DRY_RUN,
    )
    session_day = times_iter[0].date()
    cfg = replace(
        cfg,
        readiness_report=build_synthetic_readiness_report(
            checked_at_ist=times_iter[0],
            trade_date=session_day,
            instrument_keys=cfg.instrument_keys,
            cas_eligible_by_key=cfg.cas_eligible_by_key,
            tick_size_by_key=cfg.tick_size_by_key,
            exit_buffer_minutes=cfg.exit_buffer_minutes,
            approved_capital=cfg.approved_capital(),
            quote_freshness_threshold_seconds=cfg.quote_freshness_threshold_seconds,
        ),
    )
    next_time = times_iter[-1]

    def next_poll_time() -> datetime:
        nonlocal next_time
        if times_iter:
            next_time = times_iter.pop(0)
        else:
            next_time += timedelta(minutes=1)
        return next_time

    runner = ShadowLiveRunner(
        config=cfg,
        source=SyntheticQuoteSource(batches),
        now=next_poll_time,
        session_day=session_day,
    )
    runner.run()
    runner.persist(out_dir)
    return out_dir


def test_valid_profitable_session_qualifies(tmp_path: Path) -> None:
    """Prove a valid, uncorrupted profitable session qualifies as SUFFICIENT_FOR_REVIEW."""
    out = tmp_path / "valid_prof"
    _run_and_persist_session(out, strategy_name="exit-second-bar")

    policy = _make_policy()
    qualification = qualify_single_shadow_session(out, policy)

    assert (
        qualification.classification
        == ShadowEvidenceClassification.VALID_SHADOW_EVIDENCE_SUFFICIENT_FOR_REVIEW
    )
    assert qualification.qualification_passed is True
    assert qualification.qualification_reasons == ()
    assert qualification.valid_sessions_count == 1
    assert qualification.total_theoretical_trades == 1
    assert qualification.total_net_theoretical_pnl > Decimal(0)
    assert qualification.aggregate_coverage_ratio == Decimal("1.0")
    assert qualification.total_data_gaps == 0
    assert qualification.total_stale_events == 0
    assert qualification.live_orders_called is False
    assert qualification.theoretical_only_confirmation is True


def test_monday_rehearsal_qualifies_and_adapts_canonical_evidence(tmp_path: Path) -> None:
    """Prove the complete persisted Monday rehearsal reaches only paper evidence."""
    out = tmp_path / "monday_rehearsal"
    _run_and_persist_session(out, strategy_name="exit-second-bar")

    readiness = json.loads((out / "readiness-report.json").read_text(encoding="utf-8"))
    assert readiness["classification"] == "READY_FOR_RESEARCH_SHADOW"
    health = json.loads((out / "health-report.json").read_text(encoding="utf-8"))
    assert health["status"] == HealthStatus.HEALTHY.value
    assert health["live_orders_called"] is False

    qualification = qualify_single_shadow_session(out, _make_policy())
    assert (
        qualification.classification
        == ShadowEvidenceClassification.VALID_SHADOW_EVIDENCE_SUFFICIENT_FOR_REVIEW
    )
    paper_evidence = qualification.to_paper_trading_evidence()
    assert paper_evidence.environment == "shadow-live-read-only"
    assert paper_evidence.verified_orders_count == 1
    assert paper_evidence.artifact_fingerprint == qualification.qualification_fingerprint()


def test_failed_closed_health_cannot_qualify(tmp_path: Path) -> None:
    """Prove the persisted Monday health kill gate blocks evidence qualification."""
    out = tmp_path / "failed_closed"
    _run_and_persist_session(out, strategy_name="exit-second-bar")
    (out / KILL_SENTINEL).touch()

    qualification = qualify_single_shadow_session(out, _make_policy())

    assert qualification.classification == ShadowEvidenceClassification.INVALID_SHADOW_EVIDENCE
    assert qualification.qualification_passed is False
    assert any("health_failed_closed" in r for r in qualification.qualification_reasons)


def test_stale_readiness_report_cannot_qualify(tmp_path: Path) -> None:
    """Prove readiness freshness is bound to the persisted session evidence."""
    out = tmp_path / "stale_readiness"
    _run_and_persist_session(out, strategy_name="exit-second-bar")

    readiness_path = out / "readiness-report.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["checked_at_ist"] = "2026-09-07T08:00:00+05:30"
    readiness_path.write_text(
        json.dumps(readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksums = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    }
    (out / "CHECKSUMS.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
        encoding="utf-8",
    )

    qualification = qualify_single_shadow_session(out, _make_policy())

    assert qualification.classification == ShadowEvidenceClassification.INVALID_SHADOW_EVIDENCE
    assert any("readiness_stale" in r for r in qualification.qualification_reasons)


def test_missing_trade_pnl_is_unknown_not_zero(tmp_path: Path) -> None:
    """Prove absent P&L fields are not silently interpreted as zero."""
    out = tmp_path / "unknown_pnl"
    _run_and_persist_session(out, strategy_name="exit-second-bar")

    trades_path = out / "trades.jsonl"
    trade = json.loads(trades_path.read_text(encoding="utf-8").splitlines()[0])
    del trade["net_theoretical_pnl"]
    trades_path.write_text(json.dumps(trade, sort_keys=True) + "\n", encoding="utf-8")
    checksums = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    }
    (out / "CHECKSUMS.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
        encoding="utf-8",
    )

    qualification = qualify_single_shadow_session(out, _make_policy())

    assert qualification.classification == ShadowEvidenceClassification.INVALID_SHADOW_EVIDENCE
    assert any("trade_missing_pnl_evidence" in r for r in qualification.qualification_reasons)


def test_missing_qualification_policy_fails_closed(tmp_path: Path) -> None:
    """Prove qualification cannot run without an explicit, fingerprinted policy."""
    out = tmp_path / "missing_policy"
    _run_and_persist_session(out, strategy_name="exit-second-bar")

    with pytest.raises(ValueError, match="qualification policy is required"):
        qualify_single_shadow_session(out, None)  # type: ignore[arg-type]


def test_valid_losing_session_qualifies_proof_profitability_not_qualification(
    tmp_path: Path,
) -> None:
    """Prove that profitability != qualification.

    A losing session with perfect integrity and feed coverage qualifies as SUFFICIENT_FOR_REVIEW.
    Quality and validity are evaluated separately from financial performance.
    """
    out = tmp_path / "valid_losing"
    losing_batches = [
        {KEY_A: _quote(KEY_A, "2026-09-07T09:15:00+05:30", 100.0)},
        {KEY_A: _quote(KEY_A, "2026-09-07T09:20:00+05:30", 99.8)},
        {KEY_A: _quote(KEY_A, "2026-09-07T09:25:00+05:30", 99.5)},
        {KEY_A: _quote(KEY_A, "2026-09-07T09:30:00+05:30", 99.2)},
        {KEY_A: _quote(KEY_A, "2026-09-07T09:35:00+05:30", 99.0)},
    ]
    poll_times = [
        datetime.fromisoformat("2026-09-07T09:15:05+05:30"),
        datetime.fromisoformat("2026-09-07T09:20:05+05:30"),
        datetime.fromisoformat("2026-09-07T09:25:05+05:30"),
        datetime.fromisoformat("2026-09-07T09:30:05+05:30"),
        datetime.fromisoformat("2026-09-07T09:35:05+05:30"),
    ]
    _run_and_persist_session(
        out, strategy_name="exit-second-bar", batches=losing_batches, poll_times=poll_times
    )

    policy = _make_policy()
    qualification = qualify_single_shadow_session(out, policy)

    # Must qualify as SUFFICIENT_FOR_REVIEW despite negative P&L!
    assert (
        qualification.classification
        == ShadowEvidenceClassification.VALID_SHADOW_EVIDENCE_SUFFICIENT_FOR_REVIEW
    )
    assert qualification.qualification_passed is True
    assert qualification.total_theoretical_trades == 1
    assert qualification.total_net_theoretical_pnl < Decimal(0)  # Concrete proof: net loss
    assert qualification.aggregate_coverage_ratio == Decimal("1.0")


def test_checksum_corruption_fails_closed(tmp_path: Path) -> None:
    """Prove checksum corruption turns even a profitable session into INVALID_SHADOW_EVIDENCE."""
    out = tmp_path / "corrupt_checksum"
    _run_and_persist_session(out, strategy_name="exit-second-bar")

    # Tamper with summary.json
    summary_path = out / "summary.json"
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_data["tampered"] = True
    summary_path.write_text(json.dumps(summary_data, indent=2) + "\n", encoding="utf-8")

    policy = _make_policy()
    qualification = qualify_single_shadow_session(out, policy)

    assert qualification.classification == ShadowEvidenceClassification.INVALID_SHADOW_EVIDENCE
    assert qualification.qualification_passed is False
    assert any("checksum_mismatch: summary.json" in r for r in qualification.qualification_reasons)


def test_checksum_sha256_file_tampering_fails_closed(tmp_path: Path) -> None:
    """Prove tampering with CHECKSUMS.sha256 itself fails closed."""
    out = tmp_path / "tampered_checksum_file"
    _run_and_persist_session(out, strategy_name="exit-second-bar")

    # Modify the hash in CHECKSUMS.sha256
    checksums_path = out / "CHECKSUMS.sha256"
    lines = checksums_path.read_text(encoding="utf-8").splitlines()
    corrupt_lines = [("f" * 64 + "  " + line.split()[1]) for line in lines]
    checksums_path.write_text("\n".join(corrupt_lines) + "\n", encoding="utf-8")

    policy = _make_policy()
    qualification = qualify_single_shadow_session(out, policy)

    assert qualification.classification == ShadowEvidenceClassification.INVALID_SHADOW_EVIDENCE
    assert qualification.qualification_passed is False
    assert any("checksum_mismatch" in r for r in qualification.qualification_reasons)


def test_replay_mismatch_fails_closed(tmp_path: Path) -> None:
    """Prove that replayed state not matching persisted reports fails closed with INVALID."""
    out = tmp_path / "replay_mismatch"
    _run_and_persist_session(out, strategy_name="exit-second-bar")

    # Tamper with report fingerprint in report file, and update checksums so checksums pass
    # but deterministic replay verification detects the mismatch
    safe_key = KEY_A.replace("|", "_").replace(":", "_")
    rep_path = out / f"report-{safe_key}.json"
    rep_data = json.loads(rep_path.read_text(encoding="utf-8"))
    rep_data["fingerprint"] = "0" * 64  # Corrupted fingerprint
    rep_path.write_text(json.dumps(rep_data, indent=2) + "\n", encoding="utf-8")

    # Recompute CHECKSUMS.sha256 so checksum check passes, isolating replay check
    checksums = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.iterdir())
        if p.is_file() and p.name != "CHECKSUMS.sha256"
    }
    (out / "CHECKSUMS.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
        encoding="utf-8",
    )

    policy = _make_policy()
    qualification = qualify_single_shadow_session(out, policy)

    assert qualification.classification == ShadowEvidenceClassification.INVALID_SHADOW_EVIDENCE
    assert qualification.qualification_passed is False
    assert any("replay_mismatch" in r for r in qualification.qualification_reasons)


def test_insufficient_sessions_fails_closed(tmp_path: Path) -> None:
    """Prove that having fewer valid sessions than required fails closed with INSUFFICIENT."""
    out = tmp_path / "one_session"
    _run_and_persist_session(out, strategy_name="exit-second-bar")

    # Policy requires 3 valid sessions, but only 1 is provided
    policy = _make_policy(min_valid_shadow_sessions=3)
    qualification = qualify_single_shadow_session(out, policy)

    assert (
        qualification.classification
        == ShadowEvidenceClassification.VALID_SHADOW_EVIDENCE_INSUFFICIENT
    )
    assert qualification.qualification_passed is False
    assert any(
        "insufficient valid sessions: 1 < required 3" in r
        for r in qualification.qualification_reasons
    )


def test_insufficient_coverage_fails_closed(tmp_path: Path) -> None:
    """Prove that observed event coverage below policy threshold fails with INSUFFICIENT."""
    out = tmp_path / "low_coverage"
    # Max polls expected = 10, but only 3 provided (30% coverage)
    _run_and_persist_session(out, strategy_name="exit-second-bar", max_polls=10)

    # Policy requires 90% coverage
    policy = _make_policy(min_observed_coverage_ratio=Decimal("0.90"))
    qualification = qualify_single_shadow_session(out, policy)

    assert (
        qualification.classification
        == ShadowEvidenceClassification.VALID_SHADOW_EVIDENCE_INSUFFICIENT
    )
    assert qualification.qualification_passed is False
    assert any(
        "insufficient observed event coverage" in r for r in qualification.qualification_reasons
    )


def test_excessive_gap_and_stale_events_fail_closed(tmp_path: Path) -> None:
    """Prove that feed gaps or stale quotes exceeding policy thresholds fail closed."""
    # 1. Feed gap session
    gap_batches = [
        {KEY_A: _quote(KEY_A, "2026-09-07T09:15:00+05:30", 100.0)},
        {KEY_A: _quote(KEY_A, "2026-09-07T09:30:00+05:30", 101.0)},  # 15m jump = feed gap
    ]
    gap_times = [
        datetime.fromisoformat("2026-09-07T09:15:05+05:30"),
        datetime.fromisoformat("2026-09-07T09:30:05+05:30"),
    ]
    out_gap = tmp_path / "gap_session"
    _run_and_persist_session(
        out_gap,
        strategy_name="exit-second-bar",
        batches=gap_batches,
        poll_times=gap_times,
    )

    policy_no_gaps = _make_policy(
        max_feed_gap_count=0, min_theoretical_trades=0, allow_zero_trades=True
    )
    qual_gap = qualify_single_shadow_session(out_gap, policy_no_gaps)

    assert (
        qual_gap.classification == ShadowEvidenceClassification.VALID_SHADOW_EVIDENCE_INSUFFICIENT
    )
    assert qual_gap.total_data_gaps > 0
    assert any("excessive data gaps" in r for r in qual_gap.qualification_reasons)

    # 2. Stale quote session
    stale_batches = [
        {KEY_A: _quote(KEY_A, "2026-09-07T09:15:00+05:30", 100.0)},
        {KEY_A: _quote(KEY_A, "2026-09-07T09:20:00+05:30", 101.0)},
    ]
    stale_times = [
        datetime.fromisoformat("2026-09-07T09:15:05+05:30"),
        datetime.fromisoformat("2026-09-07T09:30:05+05:30"),  # Received 10 minutes late = stale
    ]
    out_stale = tmp_path / "stale_session"
    _run_and_persist_session(
        out_stale,
        strategy_name="exit-second-bar",
        batches=stale_batches,
        poll_times=stale_times,
    )

    policy_no_stale = _make_policy(
        max_stale_event_count=0, min_theoretical_trades=0, allow_zero_trades=True
    )
    qual_stale = qualify_single_shadow_session(out_stale, policy_no_stale)

    assert (
        qual_stale.classification == ShadowEvidenceClassification.VALID_SHADOW_EVIDENCE_INSUFFICIENT
    )
    assert qual_stale.total_stale_events > 0
    assert any("excessive stale events" in r for r in qual_stale.qualification_reasons)


def test_zero_trade_session_behavior(tmp_path: Path) -> None:
    """Prove zero-trade sessions can qualify under allow_zero_trades=True,

    but strictly block adaptation to PaperTradingEvidence.
    """
    out = tmp_path / "zero_trades"
    # NeverSignalShadowStrategy produces 0 trades
    _run_and_persist_session(out, strategy_name="never")

    # If policy allows zero trades
    policy_allow = _make_policy(min_theoretical_trades=0, allow_zero_trades=True)
    qual = qualify_single_shadow_session(out, policy_allow)

    assert (
        qual.classification
        == ShadowEvidenceClassification.VALID_SHADOW_EVIDENCE_SUFFICIENT_FOR_REVIEW
    )
    assert qual.total_theoretical_trades == 0

    # Adapting zero-trade evidence to PaperTradingEvidence must fail closed
    with pytest.raises(
        ZeroTradesPaperEvidenceError, match="PaperTradingEvidence requires at least one"
    ):
        qual.to_paper_trading_evidence()

    # If policy forbids zero trades
    policy_forbid = _make_policy(min_theoretical_trades=1, allow_zero_trades=False)
    qual_forbid = qualify_single_shadow_session(out, policy_forbid)

    assert (
        qual_forbid.classification
        == ShadowEvidenceClassification.VALID_SHADOW_EVIDENCE_INSUFFICIENT
    )
    assert any(
        "theoretical trade count 0 < required 1" in r for r in qual_forbid.qualification_reasons
    )


def test_cas_auxiliary_exclusion_measured(tmp_path: Path) -> None:
    """Prove CAS auxiliary quotes are measured under cas_exclusions and excluded from trades."""
    out = tmp_path / "cas_session"
    cas_batches = [
        {KEY_A: _quote(KEY_A, "2026-09-08T15:10:00+05:30", 100.0)},
        {
            KEY_A: _quote(KEY_A, "2026-09-08T15:16:00+05:30", 100.5)
        },  # At/after 15:15:00 continuous end
    ]
    cas_times = [
        datetime.fromisoformat("2026-09-08T15:10:05+05:30"),
        datetime.fromisoformat("2026-09-08T15:16:05+05:30"),
    ]
    _run_and_persist_session(
        out,
        strategy_name="always",
        batches=cas_batches,
        poll_times=cas_times,
        cas_eligible=True,
    )

    metrics = inspect_shadow_session(out)
    assert metrics.cas_exclusions >= 1
    assert "cas_auxiliary_excluded_no_trade" in metrics.no_trade_decisions_by_reason


def test_policy_mutation_changes_fingerprint(tmp_path: Path) -> None:
    """Prove changing any threshold parameter in ShadowQualificationPolicy alters fingerprints."""
    out = tmp_path / "fp_policy"
    _run_and_persist_session(out, strategy_name="exit-second-bar")

    policy_1 = _make_policy(min_observed_coverage_ratio=Decimal("0.90"))
    policy_2 = _make_policy(min_observed_coverage_ratio=Decimal("0.95"))
    policy_3 = _make_policy(max_feed_gap_count=1)

    assert policy_1.fingerprint() != policy_2.fingerprint()
    assert policy_1.fingerprint() != policy_3.fingerprint()

    qual_1 = qualify_single_shadow_session(out, policy_1)
    qual_2 = qualify_single_shadow_session(out, policy_2)

    assert qual_1.qualification_fingerprint() != qual_2.qualification_fingerprint()


def test_session_mutation_changes_fingerprint(tmp_path: Path) -> None:
    """Prove modifying any persisted session file alters session and qualification fingerprints."""
    out_1 = tmp_path / "session_orig"
    _run_and_persist_session(out_1, strategy_name="exit-second-bar")

    metrics_1 = inspect_shadow_session(out_1)

    # Different prices in session 2
    out_2 = tmp_path / "session_mutated"
    mut_batches = [
        {KEY_A: _quote(KEY_A, "2026-09-07T09:15:00+05:30", 100.0)},
        {KEY_A: _quote(KEY_A, "2026-09-07T09:20:00+05:30", 103.0)},  # Higher price
        {KEY_A: _quote(KEY_A, "2026-09-07T09:25:00+05:30", 107.0)},
        {KEY_A: _quote(KEY_A, "2026-09-07T09:30:00+05:30", 110.0)},
        {KEY_A: _quote(KEY_A, "2026-09-07T09:35:00+05:30", 114.0)},
    ]
    mut_times = [
        datetime.fromisoformat("2026-09-07T09:15:05+05:30"),
        datetime.fromisoformat("2026-09-07T09:20:05+05:30"),
        datetime.fromisoformat("2026-09-07T09:25:05+05:30"),
        datetime.fromisoformat("2026-09-07T09:30:05+05:30"),
        datetime.fromisoformat("2026-09-07T09:35:05+05:30"),
    ]
    _run_and_persist_session(
        out_2, strategy_name="exit-second-bar", batches=mut_batches, poll_times=mut_times
    )
    metrics_2 = inspect_shadow_session(out_2)

    assert metrics_1.session_fingerprint != metrics_2.session_fingerprint

    policy = _make_policy()
    qual_1 = qualify_single_shadow_session(out_1, policy)
    qual_2 = qualify_single_shadow_session(out_2, policy)

    assert qual_1.qualification_fingerprint() != qual_2.qualification_fingerprint()


def test_live_orders_called_strictly_rejected(tmp_path: Path) -> None:
    """Prove any live_orders_called=True fails closed immediately with LiveOrderAttemptError."""
    out = tmp_path / "live_orders_corrupt"
    _run_and_persist_session(out, strategy_name="exit-second-bar")

    # Inject live_orders_called=True into config.json
    cfg_path = out / "config.json"
    cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg_data["live_orders_called"] = True
    cfg_path.write_text(json.dumps(cfg_data, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(LiveOrderAttemptError, match="live orders are strictly forbidden"):
        inspect_shadow_session(out)


def test_missing_session_files_fails_closed(tmp_path: Path) -> None:
    """Prove missing files (e.g. trades.jsonl or CHECKSUMS.sha256) fail closed with INVALID."""
    out = tmp_path / "missing_files"
    _run_and_persist_session(out, strategy_name="exit-second-bar")

    # Delete trades.jsonl
    (out / "trades.jsonl").unlink()

    policy = _make_policy()
    qual = qualify_single_shadow_session(out, policy)

    assert qual.classification == ShadowEvidenceClassification.INVALID_SHADOW_EVIDENCE
    assert qual.qualification_passed is False
    assert any("missing_required_file: trades.jsonl" in r for r in qual.qualification_reasons)


def test_paper_trading_evidence_adapter_and_promotion_gate_isolation(tmp_path: Path) -> None:
    """Prove that:

    1. PaperTradingEvidence adapter works only on SUFFICIENT_FOR_REVIEW evidence.
    2. One qualified shadow session does NOT automatically promote an experiment.
       The existing promotion gate independently enforces all WFO, trade count,
       profit factor, held-out test, and provenance requirements.
    """
    out = tmp_path / "for_adapter"
    _run_and_persist_session(out, strategy_name="exit-second-bar")

    policy = _make_policy(min_theoretical_trades=1)
    qual = qualify_single_shadow_session(out, policy)

    assert (
        qual.classification
        == ShadowEvidenceClassification.VALID_SHADOW_EVIDENCE_SUFFICIENT_FOR_REVIEW
    )

    # 1. Build adapter
    paper_ev = qual.to_paper_trading_evidence()
    assert isinstance(paper_ev, PaperTradingEvidence)
    assert paper_ev.environment == "shadow-live-read-only"
    assert paper_ev.verified_orders_count == 1
    assert paper_ev.artifact_fingerprint == qual.qualification_fingerprint()

    # 2. Wire into an Experiment with promotion evidence
    orchestrator = ExperimentOrchestrator(code_commit_sha="a" * 40)
    exp = orchestrator.build_experiment(
        research_window=ResearchWindowConfig(start=date(2026, 8, 1), end=date(2026, 9, 7)),
        train_windows=(WindowSpec(1, date(2026, 8, 1), date(2026, 8, 31), 20),),
        validation_test_windows=(WindowSpec(1, date(2026, 9, 1), date(2026, 9, 7), 5),),
        embargo=EmbargoSpec(0),
        approved_capital=ApprovedCapital(Decimal(100000), "INR"),
        universe_fingerprint="u" * 64,
        candidate_prefilter_artifact_fingerprint="u" * 64,
        instrument_dataset_fingerprints={KEY_A: "a" * 64},
        nse_membership_evidence=NSEMembershipEvidenceIdentity(("ref",), True, "m" * 64, 20),
        tick_evidence=TickEvidenceIdentity("fixed-0.05", "source", True, "t" * 64),
        session_policy_identity=SessionPolicyIdentity(
            "policy", False, 15, "2026-03-01", "15:30:00"
        ),
        corporate_action_evidence=CorporateActionEvidenceIdentity("ca", True, (), "c" * 64),
        cost_model_identity=CostModelIdentity("model", "2026-09-07", {"b": "0.0003"}, ("ref",)),
        cost_evidence_identity=(
            cost_id := CostEvidenceIdentity.from_ledger(
                EffectiveDatedCostLedger(),
                on_date=date(2026, 9, 7),
                product=LedgerProduct.INTRADAY,
            )
        ),
        cost_evidence_class=cost_id.evidence_classification,
        strategy_definitions=(StrategySpec("strat_1", "ORB", "basis", (), {"buf": "5"}),),
        parameter_grid={"strat_1": {"param": ["val"]}},
        friction_scenarios=(FrictionScenarioSpec("base", Decimal(1), Decimal("0.5")),),
        rejected_candidates=(),
        tournament_result={},
        walk_forward_result={},
        promotion_evidence=ConcretePromotionEvidence(
            held_out_test=HeldOutTestEvidence(
                artifact_fingerprint="sha256_test_eval_artifact_001",
                test_dataset_fingerprints=(("NSE_EQ|INE002A01018", "fp_test_rel"),),
                window_id=1,
                trade_count=1,  # Only 1 trade from shadow session (below 20)
                profit_factor=Decimal("1.45"),
                max_drawdown_pct=Decimal("0.50"),
                drawdown_basis=DrawdownBasis.OHLC_LOW_LIQUIDATION_STRESS,
                net_return_pct=Decimal("1.20"),
                source_reference="test_window_eval_log_001",
            ),
            cost_reconciliation=CostReconciliationEvidence(
                artifact_fingerprint="sha256_cost_recon_artifact_001",
                schema_version="upstox-cost-reconciliation/v1",
                cost_model_name="documented",
                orders_checked=10,
                passed_count=10,
                failed_count=0,
                max_reconciliation_error_inr=Decimal("0.005"),
                tolerance_inr=Decimal("0.01"),
                status="PASS",
            ),
            paper_trading=paper_ev,  # Connected qualified paper trading evidence!
            baseline_comparison=BaselineComparisonEvidence(
                artifact_fingerprint="sha256_baseline_artifact_001",
                baseline_candidate_id="baseline:first-bar-hold",
                evaluated_candidate_id="strat_1",
                baseline_net_return_pct=Decimal("1.00"),
                evaluated_net_return_pct=Decimal("2.00"),
                outperformed=True,
            ),
            slippage_stress=SlippageStressEvidence(
                artifact_fingerprint="sha256_slippage_stress_artifact_001",
                scenarios_evaluated=("base_2bps", "stress_5bps"),
                stress_max_drawdown_pct=Decimal("2.00"),
                stress_passed=True,
            ),
            event_simulation=EventDrivenSimulationEvidence(
                artifact_fingerprint="sha256_event_sim_artifact_001",
                simulator_version="openalgo-event-simulator-v1",
                trade_count=10,
                initial_cash=Decimal(100000),
                final_cash=Decimal(101000),
            ),
            unpriced_cost_components=(),
        ),
    )

    # 3. Promotion gate strictly checks thresholds (e.g. min 20 trades required)
    thresholds = PromotionThresholds(
        min_trades=20,  # Requiring 20 trades
        min_profit_factor=Decimal("1.2"),
        max_drawdown_pct=Decimal("5.0"),
        min_walk_forward_windows=1,
        max_cost_reconciliation_error_inr=Decimal("0.05"),
    )
    passed, violations = exp.evaluate_promotion_gate(thresholds)

    # PROOF: One shadow session cannot promote an experiment
    assert passed is False
    assert any("trade count 1 is below required 20" in v for v in violations)
