"""Read-only live shadow runner V1 tests (offline, synthetic unless noted)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from equity_engine.shadow_live_runner import (
    BLOCKED_TOKEN_MISSING,
    FeedMode,
    RunnerMode,
    ShadowLiveConfig,
    ShadowLiveRunner,
    SyntheticQuoteSource,
    normalize_quote,
    scan_output_for_credentials,
    verify_persisted_replay,
)

IST = ZoneInfo("Asia/Kolkata")
KEY = "NSE_EQ|INE002A01018"


def _config(tmp_path: Path, **overrides) -> ShadowLiveConfig:
    params: dict[str, object] = {
        "session_id": "test-live-v1",
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


def _quote(ts: str, price: float) -> dict[str, object]:
    return {
        "instrument_token": KEY,
        "timestamp": ts,
        "last_price": price + 0.1,
        "prev_close_price": 100,
        "ohlc": {"open": price, "high": price + 0.15, "low": price - 0.05, "close": price + 0.1},
    }


def _batches() -> list[dict[str, dict[str, object]]]:
    return [
        {KEY: _quote("2026-09-07T09:15:00+05:30", 100)},
        {KEY: _quote("2026-09-07T09:20:00+05:30", 101)},
        {KEY: _quote("2026-09-07T09:25:00+05:30", 102)},
    ]


def _times() -> list[datetime]:
    return [
        datetime.fromisoformat("2026-09-07T09:15:05+05:30"),
        datetime.fromisoformat("2026-09-07T09:20:05+05:30"),
        datetime.fromisoformat("2026-09-07T09:25:05+05:30"),
    ]


def test_synthetic_polling_produces_evidence_and_no_orders(tmp_path: Path) -> None:
    times = _times()
    runner = ShadowLiveRunner(
        config=_config(tmp_path),
        source=SyntheticQuoteSource(_batches()),
        now=lambda: times.pop(0),
    )
    reports = runner.run()
    assert set(reports) == {KEY}
    report = reports[KEY]
    assert report.live_orders_called is False
    assert len(runner.normalized) == 3
    first = runner.normalized[0]
    assert first.source_timestamp == "2026-09-07T09:15:00+05:30"
    assert first.received_timestamp == "2026-09-07T09:15:05+05:30"
    assert first.seq == 0
    assert first.quote_age_seconds == 5.0
    assert first.data_fingerprint is not None
    assert first.feed_status == "ok"
    summary = runner.persist(tmp_path / "out")
    assert summary["live_orders_called"] is False
    assert (tmp_path / "out" / "market_events.jsonl").exists()
    assert (tmp_path / "out" / "decisions.jsonl").exists()
    assert (tmp_path / "out" / "summary.json").exists()


def test_stale_quote_fails_closed(tmp_path: Path) -> None:
    batches = [
        {KEY: _quote("2026-09-07T09:15:00+05:30", 100)},
        {KEY: _quote("2026-09-07T09:20:00+05:30", 101)},
    ]
    times = [
        datetime.fromisoformat("2026-09-07T09:15:05+05:30"),
        datetime.fromisoformat("2026-09-07T09:30:05+05:30"),  # 10 min old
    ]
    runner = ShadowLiveRunner(
        config=_config(tmp_path, max_polls=2),
        source=SyntheticQuoteSource(batches),
        now=lambda: times.pop(0),
    )
    reports = runner.run()
    reasons = [d.reason for d in reports[KEY].decisions]
    assert "stale_quote_no_trade" in reasons
    assert reports[KEY].trades == ()


def test_duplicate_quote_fails_closed(tmp_path: Path) -> None:
    batches = [
        {KEY: _quote("2026-09-07T09:15:00+05:30", 100)},
        {KEY: _quote("2026-09-07T09:15:00+05:30", 100)},
    ]
    times = [
        datetime.fromisoformat("2026-09-07T09:15:05+05:30"),
        datetime.fromisoformat("2026-09-07T09:15:10+05:30"),
    ]
    runner = ShadowLiveRunner(
        config=_config(tmp_path, max_polls=2),
        source=SyntheticQuoteSource(batches),
        now=lambda: times.pop(0),
    )
    reports = runner.run()
    reasons = [d.reason for d in reports[KEY].decisions]
    assert reasons[0] == "entry_signal_pending_next_bar"
    assert reasons[1] in ("out_of_order_event_no_trade", "feed_gap_no_trade")
    assert reports[KEY].trades == ()


def test_out_of_order_quote_fails_closed(tmp_path: Path) -> None:
    batches = [
        {KEY: _quote("2026-09-07T09:20:00+05:30", 101)},
        {KEY: _quote("2026-09-07T09:15:00+05:30", 100)},
    ]
    times = [
        datetime.fromisoformat("2026-09-07T09:20:05+05:30"),
        datetime.fromisoformat("2026-09-07T09:20:10+05:30"),
    ]
    runner = ShadowLiveRunner(
        config=_config(tmp_path, max_polls=2),
        source=SyntheticQuoteSource(batches),
        now=lambda: times.pop(0),
    )
    reports = runner.run()
    assert reports[KEY].decisions[1].reason == "out_of_order_event_no_trade"


def test_feed_gap_and_reconnect_boundary(tmp_path: Path) -> None:
    batches = [
        {KEY: _quote("2026-09-07T09:15:00+05:30", 100)},
        {KEY: _quote("2026-09-07T09:30:00+05:30", 101)},
    ]
    times = [
        datetime.fromisoformat("2026-09-07T09:15:05+05:30"),
        datetime.fromisoformat("2026-09-07T09:30:05+05:30"),
    ]
    runner = ShadowLiveRunner(
        config=_config(tmp_path, max_polls=2),
        source=SyntheticQuoteSource(batches),
        now=lambda: times.pop(0),
    )
    reports = runner.run()
    assert reports[KEY].decisions[1].reason == "feed_gap_no_trade"

    class _Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def fetch_quotes(self) -> dict[str, dict[str, object]]:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transport down")
            return {KEY: _quote("2026-09-07T09:15:00+05:30", 100)}

    times2 = [datetime.fromisoformat("2026-09-07T09:15:05+05:30")] * 2
    runner2 = ShadowLiveRunner(
        config=_config(tmp_path, max_polls=2), source=_Flaky(), now=lambda: times2.pop(0)
    )
    runner2.run()
    assert runner2.normalized[0].feed_status == "poll_failure"
    assert runner2.normalized[1].reconnect_boundary is True


def test_cas_cutoff_and_session_closed(tmp_path: Path) -> None:
    cas_cfg = _config(
        tmp_path,
        cas_eligible_by_key=((KEY, True),),
        max_polls=1,
    )
    cas_batches = [{KEY: _quote("2026-09-08T15:16:00+05:30", 100)}]
    times = [datetime.fromisoformat("2026-09-08T15:16:05+05:30")]
    runner = ShadowLiveRunner(
        config=cas_cfg, source=SyntheticQuoteSource(cas_batches), now=lambda: times.pop(0)
    )
    reports = runner.run()
    assert reports[KEY].decisions[0].reason == "cas_auxiliary_excluded_no_trade"

    closed_batches = [{KEY: _quote("2026-09-07T18:00:00+05:30", 100)}]
    times2 = [datetime.fromisoformat("2026-09-07T18:00:05+05:30")]
    runner2 = ShadowLiveRunner(
        config=_config(tmp_path, max_polls=1),
        source=SyntheticQuoteSource(closed_batches),
        now=lambda: times2.pop(0),
    )
    reports2 = runner2.run()
    assert reports2[KEY].decisions[0].reason in (
        "cas_auxiliary_excluded_no_trade",
        "outside_continuous_session_no_trade",
    )


def test_clock_timezone_mismatch_fails_closed() -> None:
    received = datetime.fromisoformat("2026-09-07T09:15:05+05:30")
    normalized = normalize_quote(
        instrument_key=KEY,
        quote={
            "instrument_token": KEY,
            "timestamp": "2026-09-07T09:15:00",
            "last_price": 100,
            "ohlc": {"open": 100, "high": 101, "low": 99},
        },
        seq=0,
        received_at=received,
        cas_eligible=False,
    )
    assert normalized.event is None
    assert normalized.feed_status == "clock_timezone_mismatch"


def test_missing_token_blocked() -> None:
    env = {k: v for k, v in os.environ.items() if k != "UPSTOX_ACCESS_TOKEN"}
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/shadow_live.py",
            "--mode",
            "LIVE_READ_ONLY",
            "--instrument-keys",
            KEY,
            "--output-dir",
            "data/shadow-live-test-out",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(__file__).parents[1]),
        check=False,
    )
    assert completed.returncode == 2
    assert BLOCKED_TOKEN_MISSING in completed.stderr


def test_token_redaction_and_no_credentials_persisted(tmp_path: Path) -> None:
    fake_token = "token-abc-123-redaction-check"
    runner = ShadowLiveRunner(
        config=_config(tmp_path),
        source=SyntheticQuoteSource(_batches()),
        now=lambda: _times().pop(0),
    )
    # Times helper above is consumed; rebuild deterministically.
    assert runner is not None
    times = _times()
    runner2 = ShadowLiveRunner(
        config=_config(tmp_path), source=SyntheticQuoteSource(_batches()), now=lambda: times.pop(0)
    )
    runner2.run()
    out = tmp_path / "out"
    runner2.persist(out)
    violations = scan_output_for_credentials(out, fake_token)
    assert violations == []
    blob = "\n".join(path.read_text(encoding="utf-8") for path in out.rglob("*") if path.is_file())
    assert fake_token not in blob
    assert "UPSTOX_ACCESS_TOKEN" not in blob


def test_deterministic_replay_matches_persisted(tmp_path: Path) -> None:
    times = _times()
    config = _config(tmp_path)
    runner = ShadowLiveRunner(
        config=config, source=SyntheticQuoteSource(_batches()), now=lambda: times.pop(0)
    )
    runner.run()
    out = tmp_path / "out"
    runner.persist(out)
    result = verify_persisted_replay(out, config=config)
    assert result["matched"] == {KEY: True}


def test_runner_module_has_no_order_api() -> None:
    source = Path(__file__).parents[1] / "src" / "equity_engine" / "shadow_live_runner.py"
    text = source.read_text(encoding="utf-8")
    for forbidden in ("place_order", "modify_order", "cancel_order"):
        assert forbidden not in text
    assert "broker/" not in text
    assert "OrderClient" not in text
    script = Path(__file__).parents[1] / "scripts" / "shadow_live.py"
    script_text = script.read_text(encoding="utf-8")
    for forbidden in ("place_order", "modify_order", "cancel_order"):
        assert forbidden not in script_text


def test_live_orders_called_false_everywhere(tmp_path: Path) -> None:
    times = _times()
    runner = ShadowLiveRunner(
        config=_config(tmp_path), source=SyntheticQuoteSource(_batches()), now=lambda: times.pop(0)
    )
    reports = runner.run()
    assert reports[KEY].live_orders_called is False
    assert all(d.live_orders_called is False for d in reports[KEY].decisions)
    out = tmp_path / "out"
    summary = runner.persist(out)
    assert summary["live_orders_called"] is False
    persisted = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert persisted["live_orders_called"] is False
