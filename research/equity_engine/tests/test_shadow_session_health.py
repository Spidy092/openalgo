"""Shadow session health monitor V1 tests (offline, synthetic)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from equity_engine.shadow_live_runner import (
    FeedMode,
    RunnerMode,
    ShadowLiveConfig,
    ShadowLiveRunner,
    SyntheticQuoteSource,
)
from equity_engine.shadow_session_health import (
    HealthStatus,
    HealthThresholds,
    check_persisted_session,
    evaluate_session_health,
)

KEY = "NSE_EQ|INE002A01018"
NOW = "2026-09-07T09:25:10+05:30"


def _quote(ts: str, price: float) -> dict[str, object]:
    return {
        "instrument_token": KEY,
        "timestamp": ts,
        "last_price": price + 0.1,
        "prev_close_price": 100,
        "ohlc": {"open": price, "high": price + 0.15, "low": price - 0.05, "close": price + 0.1},
    }


def _config(tmp_path: Path, **overrides) -> ShadowLiveConfig:
    params: dict[str, object] = {
        "session_id": "health-v1",
        "instrument_keys": (KEY,),
        "cas_eligible_by_key": ((KEY, False),),
        "tick_size_by_key": ((KEY, "0.05"),),
        "feed_mode": FeedMode.POLL,
        "poll_interval_seconds": 0.0,
        "max_polls": 3,
        "quote_freshness_threshold_seconds": 60.0,
        "expected_cadence_seconds": 300.0,
        "approved_capital_rupees": "100000",
        "exit_buffer_minutes": 15,
        "strategy_name": "always",
        "output_dir": str(tmp_path / "out"),
        "mode": RunnerMode.DRY_RUN,
    }
    params.update(overrides)
    return ShadowLiveConfig(**params)  # type: ignore[arg-type]


def _run_session(
    tmp_path: Path, batches: list[dict[str, dict[str, object]]], times: list[str]
) -> Path:
    from datetime import datetime

    moments = [datetime.fromisoformat(item) for item in times]
    runner = ShadowLiveRunner(
        config=_config(tmp_path, max_polls=len(batches)),
        source=SyntheticQuoteSource(batches),
        now=lambda: moments.pop(0),
    )
    runner.run()
    out = tmp_path / "out"
    runner.persist(out)
    return out


def test_healthy_session_allows_trades(tmp_path: Path) -> None:
    out = _run_session(
        tmp_path,
        [
            {KEY: _quote("2026-09-07T09:15:00+05:30", 100)},
            {KEY: _quote("2026-09-07T09:20:00+05:30", 101)},
            {KEY: _quote("2026-09-07T09:25:00+05:30", 102)},
        ],
        [
            "2026-09-07T09:15:05+05:30",
            "2026-09-07T09:20:05+05:30",
            "2026-09-07T09:25:05+05:30",
        ],
    )
    report = check_persisted_session(out, now_iso=NOW)
    assert report.status is HealthStatus.HEALTHY
    assert report.allow_new_theoretical_trades is True
    assert report.runner_started and report.runner_stopped
    assert report.decisions_count == 3
    assert report.live_orders_called is False
    assert report.persistence_ok and report.checksums_ok


def test_stale_feed_degrades_and_blocks_trades() -> None:
    report = evaluate_session_health(
        session_id="s",
        runner_started=True,
        runner_stopped=False,
        killed=False,
        disk_error=False,
        persistence_ok=True,
        checksums_ok=True,
        reports_ok=True,
        normalized_feed_statuses=("ok", "ok"),
        normalized_received_ats=("2026-09-07T09:15:05+05:30", "2026-09-07T09:20:05+05:30"),
        decision_reasons=(),
        last_quote_age_seconds=600.0,
        now_iso=NOW,
        thresholds=HealthThresholds(freshness_seconds=60.0),
    )
    assert report.status is HealthStatus.DEGRADED_NO_TRADING
    assert report.allow_new_theoretical_trades is False
    assert any("stale" in reason for reason in report.reasons)


def test_gap_requires_explicit_recovery_and_is_not_hidden(tmp_path: Path) -> None:
    out = _run_session(
        tmp_path,
        [
            {KEY: _quote("2026-09-07T09:15:00+05:30", 100)},
            {KEY: _quote("2026-09-07T09:30:00+05:30", 101)},
        ],
        ["2026-09-07T09:15:05+05:30", "2026-09-07T09:30:05+05:30"],
    )
    report = check_persisted_session(out, now_iso="2026-09-07T09:30:10+05:30")
    assert report.status is HealthStatus.DEGRADED_NO_TRADING
    assert report.feed_gaps == 1
    assert any("gap" in reason for reason in report.reasons)


def test_reconnect_duplicate_out_of_order_degrade() -> None:
    report = evaluate_session_health(
        session_id="s",
        runner_started=True,
        runner_stopped=True,
        killed=False,
        disk_error=False,
        persistence_ok=True,
        checksums_ok=True,
        reports_ok=True,
        normalized_feed_statuses=("ok", "poll_failure", "ok"),
        normalized_received_ats=(
            "2026-09-07T09:15:05+05:30",
            "2026-09-07T09:20:05+05:30",
            "2026-09-07T09:25:05+05:30",
        ),
        normalized_reconnects=(False, False, True),
        decision_reasons=("duplicate_event_no_trade", "out_of_order_event_no_trade"),
        now_iso=NOW,
    )
    assert report.status is HealthStatus.DEGRADED_NO_TRADING
    assert report.allow_new_theoretical_trades is False


def test_cas_uncertainty_blocks_trades(tmp_path: Path) -> None:
    config = _config(tmp_path, cas_eligible_by_key=((KEY, True),), max_polls=1)
    from datetime import datetime

    moments = [datetime.fromisoformat("2026-09-08T15:16:05+05:30")]
    runner = ShadowLiveRunner(
        config=config,
        source=SyntheticQuoteSource([{KEY: _quote("2026-09-08T15:16:00+05:30", 100)}]),
        now=lambda: moments.pop(0),
    )
    runner.run()
    out = tmp_path / "out"
    runner.persist(out)
    report = check_persisted_session(out, now_iso="2026-09-08T15:16:10+05:30")
    assert report.status is HealthStatus.DEGRADED_NO_TRADING
    assert report.cas_auxiliary_events >= 1
    assert report.allow_new_theoretical_trades is False


def test_checksum_corruption_fails_closed(tmp_path: Path) -> None:
    out = _run_session(
        tmp_path,
        [
            {KEY: _quote("2026-09-07T09:15:00+05:30", 100)},
            {KEY: _quote("2026-09-07T09:20:00+05:30", 101)},
            {KEY: _quote("2026-09-07T09:25:00+05:30", 102)},
        ],
        [
            "2026-09-07T09:15:05+05:30",
            "2026-09-07T09:20:05+05:30",
            "2026-09-07T09:25:05+05:30",
        ],
    )
    decisions = out / "decisions.jsonl"
    decisions.write_text(
        decisions.read_text(encoding="utf-8") + '{"tampered": true}\n', encoding="utf-8"
    )
    report = check_persisted_session(out, now_iso=NOW)
    assert report.status is HealthStatus.FAILED_CLOSED
    assert report.allow_new_theoretical_trades is False
    assert any("checksum" in reason for reason in report.reasons)


def test_missing_files_and_kill_and_disk_fail_closed(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    report = check_persisted_session(out, now_iso=NOW)
    assert report.status is HealthStatus.FAILED_CLOSED

    healthy = _run_session(
        tmp_path,
        [{KEY: _quote("2026-09-07T09:15:00+05:30", 100)}],
        ["2026-09-07T09:15:05+05:30"],
    )
    (healthy / "KILL").write_text("stop\n", encoding="utf-8")
    killed = check_persisted_session(healthy, now_iso="2026-09-07T09:15:10+05:30")
    assert killed.status is HealthStatus.FAILED_CLOSED
    assert killed.killed is True

    (healthy / "KILL").unlink()
    (healthy / "DISK_ERROR").write_text("write failed\n", encoding="utf-8")
    disk = check_persisted_session(healthy, now_iso="2026-09-07T09:15:10+05:30")
    assert disk.status is HealthStatus.FAILED_CLOSED
    assert disk.disk_error is True


def test_health_report_deterministic_fingerprint(tmp_path: Path) -> None:
    out = _run_session(
        tmp_path,
        [
            {KEY: _quote("2026-09-07T09:15:00+05:30", 100)},
            {KEY: _quote("2026-09-07T09:20:00+05:30", 101)},
            {KEY: _quote("2026-09-07T09:25:00+05:30", 102)},
        ],
        [
            "2026-09-07T09:15:05+05:30",
            "2026-09-07T09:20:05+05:30",
            "2026-09-07T09:25:05+05:30",
        ],
    )
    first = check_persisted_session(out, now_iso=NOW)
    second = check_persisted_session(out, now_iso=NOW)
    assert first.fingerprint() == second.fingerprint()
    assert first.as_dict() == second.as_dict()


def test_cli_status_command_reports_healthy(tmp_path: Path) -> None:
    out = _run_session(
        tmp_path,
        [
            {KEY: _quote("2026-09-07T09:15:00+05:30", 100)},
            {KEY: _quote("2026-09-07T09:20:00+05:30", 101)},
            {KEY: _quote("2026-09-07T09:25:00+05:30", 102)},
        ],
        [
            "2026-09-07T09:15:05+05:30",
            "2026-09-07T09:20:05+05:30",
            "2026-09-07T09:25:05+05:30",
        ],
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shadow_health.py",
            "--output-dir",
            str(out),
            "--now",
            NOW,
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parents[1]),
        check=False,
    )
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "HEALTHY"
    assert payload["live_orders_called"] is False


def test_health_module_has_no_order_api() -> None:
    for name in ("shadow_session_health.py",):
        text = (Path(__file__).parents[1] / "src" / "equity_engine" / name).read_text(
            encoding="utf-8"
        )
        for forbidden in ("place_order", "modify_order", "cancel_order"):
            assert forbidden not in text
        assert "OrderClient" not in text
    script_text = (Path(__file__).parents[1] / "scripts" / "shadow_health.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("place_order", "modify_order", "cancel_order"):
        assert forbidden not in script_text


def test_live_orders_called_false() -> None:
    report = evaluate_session_health(
        session_id="s",
        runner_started=True,
        runner_stopped=True,
        killed=False,
        disk_error=False,
        persistence_ok=True,
        checksums_ok=True,
        reports_ok=True,
        normalized_feed_statuses=("ok",),
        normalized_received_ats=("2026-09-07T09:15:05+05:30",),
        now_iso="2026-09-07T09:15:10+05:30",
    )
    assert report.live_orders_called is False
    assert report.as_dict()["live_orders_called"] is False
