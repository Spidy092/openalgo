import importlib.util
import json
from dataclasses import asdict
from datetime import datetime, time, timezone
from pathlib import Path

import pandas as pd

from equity_engine.historical_validation import (
    IntradaySessionRule,
    nse_session_rules_for_calendar,
    validate_intraday_dataset,
)
from equity_engine.nse_calendar import nse_cm_normal_session_calendar
from equity_engine.provenance import MarketDataManifest, dataframe_fingerprint


def _frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    rows = len(index)
    return pd.DataFrame(
        {
            "open": [100.0 + item for item in range(rows)],
            "high": [101.0 + item for item in range(rows)],
            "low": [99.0 + item for item in range(rows)],
            "close": [100.5 + item for item in range(rows)],
            "volume": [1000 + item for item in range(rows)],
        },
        index=index,
    )


def _manifest(frame: pd.DataFrame) -> MarketDataManifest:
    return MarketDataManifest(
        provider="unit-test",
        exchange="NSE",
        instrument_token="NSE_EQ|TEST",
        symbol="TEST",
        timezone="Asia/Kolkata",
        interval="5m",
        timestamp_semantics="candle_start",
        start=frame.index[0].to_pydatetime(),
        end=frame.index[-1].to_pydatetime(),
        retrieved_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
        adjustment_policy="none",
        universe_rule_version="test-v1",
        source_reference="unit-test",
    )


def _rule() -> IntradaySessionRule:
    return IntradaySessionRule(
        rule_id="test-session",
        timezone="Asia/Kolkata",
        start_time=time(9, 15),
        end_time=time(9, 25),
        interval_minutes=5,
        source_reference="unit-test",
    )


def test_complete_day_reports_explicit_coverage() -> None:
    index = pd.date_range("2026-09-07 09:15", periods=2, freq="5min", tz="Asia/Kolkata")
    frame = _frame(index)
    manifest = _manifest(frame)

    report = validate_intraday_dataset(
        frame,
        manifest,
        session_rules={index[0].date(): _rule()},
        manifest_fingerprint_reference=dataframe_fingerprint(frame, manifest),
    )

    assert report.passed
    assert report.per_day[0].row_count == 2
    assert report.per_day[0].missing_expected_slots == ()
    assert report.per_day[0].unexpected_timestamps == ()


def test_missing_and_unexpected_slots_are_reported() -> None:
    index = pd.DatetimeIndex(["2026-09-07 09:15", "2026-09-07 09:25"], tz="Asia/Kolkata")
    frame = _frame(index)
    manifest = _manifest(frame)

    report = validate_intraday_dataset(
        frame,
        manifest,
        session_rules={index[0].date(): _rule()},
        manifest_fingerprint_reference=dataframe_fingerprint(frame, manifest),
    )

    assert not report.passed
    assert report.per_day[0].missing_expected_slots == ("2026-09-07 09:20:00+05:30",)
    assert report.per_day[0].unexpected_timestamps == ("2026-09-07 09:25:00+05:30",)


def test_undeclared_date_fails_closed_instead_of_assuming_normal_session() -> None:
    index = pd.date_range("2026-09-07 09:15", periods=1, freq="5min", tz="Asia/Kolkata")
    frame = _frame(index)
    manifest = _manifest(frame)

    report = validate_intraday_dataset(
        frame,
        manifest,
        session_rules={},
        manifest_fingerprint_reference=dataframe_fingerprint(frame, manifest),
    )

    assert not report.passed
    assert any("without an explicit session rule" in item for item in report.structural_violations)
    assert report.per_day[0].session_rule is None


def test_calendar_rules_keep_cas_effective_date_explicit() -> None:
    calendar = nse_cm_normal_session_calendar(
        start=datetime(2026, 8, 4).date(), end=datetime(2026, 8, 4).date()
    )

    rules = nse_session_rules_for_calendar(
        calendar,
        timezone="Asia/Kolkata",
        interval_minutes=5,
        cas_eligible=True,
    )

    assert rules[datetime(2026, 8, 4).date()].rule_id == "nse-cm-cas-continuous-session"
    assert rules[datetime(2026, 8, 4).date()].end_time == time(15, 15)


def test_offline_validation_script_reads_artifact_without_network(tmp_path, capsys) -> None:
    index = pd.date_range("2026-09-07 09:15", periods=75, freq="5min", tz="Asia/Kolkata")
    frame = _frame(index)
    manifest = _manifest(frame)
    parquet = tmp_path / "pilot.parquet"
    manifest_path = tmp_path / "pilot.manifest.json"
    frame.to_parquet(parquet)
    manifest_path.write_text(
        json.dumps(
            {
                "instrument": {"cas_eligible": False},
                "dataset_manifest": asdict(manifest),
                "fingerprint_sha256": dataframe_fingerprint(frame, manifest),
            },
            default=str,
        ),
        encoding="utf-8",
    )

    script_path = Path(__file__).parents[1] / "scripts" / "validate_historical_dataset.py"
    spec = importlib.util.spec_from_file_location("validate_historical_dataset_cli", script_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    status = cli.main(["--parquet", str(parquet), "--manifest", str(manifest_path)])

    assert status == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "PASS"
    assert output["rows"] == 75
    assert output["strategy_ready"] is False
    assert output["live_orders_called"] is False
