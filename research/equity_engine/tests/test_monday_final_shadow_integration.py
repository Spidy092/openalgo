"""Monday V2 contract tests: canonical readiness into read-only shadow."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from equity_engine.experiment import CostReconciliationEvidence
from equity_engine.historical_validation import (
    DailyIntradayValidation,
    HistoricalDatasetValidation,
)
from equity_engine.live_market_readiness import (
    CAS_POLICY_UNKNOWN,
    FEED_GAP_DETECTED,
    FEED_UNAVAILABLE,
    QUOTE_STALE,
    FeedHealthEvidence,
    LiveMarketReadinessReport,
    ReadinessClassification,
    ReasonDetail,
    build_synthetic_readiness_report,
)
from equity_engine.monday_readiness_probe import (
    MondayProbeConfig,
    ProbeMode,
    run_live_read_only,
)
from equity_engine.shadow_live_runner import (
    READINESS_CONTEXT_MISMATCH,
    READINESS_NOT_READY,
    READINESS_REPORT_MISSING,
    READINESS_STALE,
    FeedMode,
    ReadinessGateError,
    RunnerMode,
    ShadowLiveConfig,
    ShadowLiveRunner,
    SyntheticQuoteSource,
    scan_output_for_credentials,
    verify_persisted_replay,
)
from equity_engine.upstox_market_context import QuoteBatchResult
from equity_engine.upstox_readiness import ReadinessCheck, UpstoxReadinessSnapshot

IST = ZoneInfo("Asia/Kolkata")
TRADE_DATE = date(2026, 9, 7)
KEY = "NSE_EQ|INE002A01018"


def _quote(timestamp: datetime, price: int = 100) -> dict[str, object]:
    return {
        "instrument_token": KEY,
        "timestamp": timestamp.isoformat(),
        "last_price": price + 0.1,
        "prev_close_price": 100,
        "ohlc": {
            "open": price,
            "high": price + 0.15,
            "low": price - 0.05,
            "close": price + 0.1,
        },
    }


def _config(
    tmp_path: Path,
    *,
    classification: ReadinessClassification = ReadinessClassification.READY_FOR_RESEARCH_SHADOW,
    checked_at: datetime | None = None,
    trade_date: date = TRADE_DATE,
    capital: str = "100000",
    cas_eligible: bool = False,
    exit_buffer_minutes: int = 15,
    max_polls: int = 3,
) -> ShadowLiveConfig:
    config = ShadowLiveConfig(
        session_id="monday-final-shadow-v2",
        instrument_keys=(KEY,),
        cas_eligible_by_key=((KEY, cas_eligible),),
        tick_size_by_key=((KEY, "0.05"),),
        feed_mode=FeedMode.POLL,
        poll_interval_seconds=0.0,
        max_polls=max_polls,
        quote_freshness_threshold_seconds=60.0,
        expected_cadence_seconds=300.0,
        approved_capital_rupees=capital,
        exit_buffer_minutes=exit_buffer_minutes,
        strategy_name="always",
        output_dir=str(tmp_path / "shadow"),
        mode=RunnerMode.LIVE_READ_ONLY,
    )
    report = build_synthetic_readiness_report(
        checked_at_ist=checked_at or datetime(2026, 9, 7, 9, 15, 5, tzinfo=IST),
        trade_date=trade_date,
        instrument_keys=config.instrument_keys,
        cas_eligible_by_key=config.cas_eligible_by_key,
        tick_size_by_key=config.tick_size_by_key,
        exit_buffer_minutes=config.exit_buffer_minutes,
        approved_capital=config.approved_capital(),
        quote_freshness_threshold_seconds=config.quote_freshness_threshold_seconds,
        classification=classification,
    )
    return replace(config, readiness_report=report)


def _with_reason(config: ShadowLiveConfig, code: str) -> ShadowLiveConfig:
    report = config.readiness_report
    assert report is not None
    reason = ReasonDetail(
        code=code,
        message=code,
        blocks_infra=True,
        blocks_research_shadow=True,
    )
    return replace(
        config,
        readiness_report=replace(
            report,
            classification=ReadinessClassification.NOT_READY_FOR_SHADOW,
            reason_codes=(code,),
            reasons=(reason,),
        ),
    )


class _CountingSource(SyntheticQuoteSource):
    def __init__(self, batches: list[dict[str, dict[str, object]]]) -> None:
        super().__init__(batches)
        self.calls = 0

    def fetch_quotes(self) -> dict[str, dict[str, object]]:
        self.calls += 1
        return super().fetch_quotes()


def _normal_batches(
    *, start: datetime | None = None, count: int = 3
) -> tuple[list[dict[str, dict[str, object]]], list[datetime]]:
    first = start or datetime(2026, 9, 7, 9, 15, tzinfo=IST)
    batches = [
        {KEY: _quote(first + timedelta(minutes=5 * index), 100 + index)} for index in range(count)
    ]
    times = [first + timedelta(minutes=5 * index, seconds=5) for index in range(count)]
    return batches, times


def _probe_bod_row() -> dict[str, object]:
    return {
        "segment": "NSE_EQ",
        "name": "RELIANCE INDUSTRIES",
        "exchange": "NSE",
        "isin": "INE002A01018",
        "instrument_type": "EQ",
        "instrument_key": KEY,
        "exchange_token": 2885,
        "lot_size": 1,
        "freeze_quantity": 100000,
        "tick_size": 5,
        "trading_symbol": "RELIANCE",
        "series": "EQ",
        "security_type": "NORMAL",
        "cas_eligible": False,
    }


def _probe_historical() -> HistoricalDatasetValidation:
    per_day = DailyIntradayValidation(
        trade_date=TRADE_DATE,
        row_count=2,
        first_timestamp="2026-09-07 09:15:00+05:30",
        last_timestamp="2026-09-07 09:20:00+05:30",
        duplicate_count=0,
        continuous_session_rows=2,
        cas_auxiliary_rows=0,
        cas_auxiliary_timestamps=(),
        missing_expected_slots=(),
        unexpected_timestamps=(),
        timezone="Asia/Kolkata",
        ohlcv_violations=(),
        session_rule={"rule_id": "synthetic-normal-session"},
    )
    return HistoricalDatasetValidation(
        rows=2,
        trading_dates=(TRADE_DATE,),
        per_day=(per_day,),
        structural_violations=(),
        deterministic_data_fingerprint="d" * 64,
        manifest_fingerprint_reference="m" * 64,
        fingerprint_schema="equity-market-data-v2",
        manifest_reference="synthetic-validated-dataset",
        timezone="Asia/Kolkata",
        calendar_evidence=None,
    )


def _probe_quote(timestamp: datetime) -> QuoteBatchResult:
    return QuoteBatchResult(
        requested_instrument_keys=(KEY,),
        quotes={
            KEY: {
                "instrument_token": KEY,
                "timestamp": timestamp.isoformat(),
                "last_price": "100",
                "prev_close_price": "100",
                "ohlc": {"open": "100", "high": "100.15", "low": "99.95", "close": "100.1"},
            }
        },
        failures={},
        request_count=1,
    )


def _probe_readiness_report(checked_at: datetime):
    config = MondayProbeConfig(
        mode=ProbeMode.LIVE_READ_ONLY,
        instrument_key=KEY,
        approved_capital_rupees=Decimal(100000),
        cost_tolerance_inr=Decimal("0.01"),
        max_quote_age_seconds=60.0,
        exit_buffer_minutes=59,
        tick_size_scale_rupees_per_raw_unit=Decimal("0.01"),
        tick_reference_price_rupees=Decimal(500),
        pit_complete=True,
        historical_validation=_probe_historical(),
        strategy_evidence_present=True,
        cost_evidence=CostReconciliationEvidence(
            artifact_fingerprint="cost-artifact-synthetic",
            schema_version="upstox-cost-reconciliation/v1",
            cost_model_name="documented",
            orders_checked=1,
            passed_count=1,
            failed_count=0,
            max_reconciliation_error_inr=Decimal("0.00"),
            tolerance_inr=Decimal("0.01"),
            status="PASS",
        ),
        paper_evidence=None,
        feed=FeedHealthEvidence(
            available=True,
            gap_detected=False,
            last_heartbeat_ist=checked_at,
        ),
        kill_switch_engaged=False,
    )
    snapshot = UpstoxReadinessSnapshot(
        checks=tuple(
            ReadinessCheck(name=name, passed=True, detail="synthetic")
            for name in (
                "profile_api",
                "nse_enabled",
                "intraday_product_enabled",
                "funds_api",
                "minimum_test_capital_present",
                "primary_static_ip",
            )
        ),
        available_to_trade=Decimal(150000),
        exchanges=("NSE",),
        products=("D",),
        primary_static_ip_configured=True,
        secondary_static_ip_configured=False,
    )
    result = run_live_read_only(
        config,
        token_present=True,
        now_ist=checked_at,
        readiness_snapshot=snapshot,
        quotes=_probe_quote(checked_at - timedelta(seconds=5)),
        bod_rows=(_probe_bod_row(),),
        mis_rows=({"instrument_key": KEY},),
        suspended_rows=(),
    )
    assert result.report is not None
    return result


def _runner(
    config: ShadowLiveConfig,
    batches: list[dict[str, dict[str, object]]],
    times: list[datetime],
) -> ShadowLiveRunner:
    return ShadowLiveRunner(
        config=config,
        source=SyntheticQuoteSource(batches),
        now=iter(times).__next__,
        session_day=TRADE_DATE,
    )


@pytest.mark.parametrize(
    ("classification", "strategy_enabled"),
    [
        (ReadinessClassification.READY_FOR_SHADOW_INFRA, False),
        (ReadinessClassification.READY_FOR_RESEARCH_SHADOW, True),
        (ReadinessClassification.READY_FOR_LIVE_ORDER_REVIEW, True),
    ],
)
def test_readiness_classification_controls_strategy_execution(
    tmp_path: Path,
    classification: ReadinessClassification,
    strategy_enabled: bool,
) -> None:
    config = _config(tmp_path, classification=classification)
    batches, times = _normal_batches()
    runner = _runner(config, batches, times)

    reports = runner.run()

    assert runner._shadow_strategy_enabled is strategy_enabled
    assert all(item.event is not None for item in runner.normalized)
    assert all(report.live_orders_called is False for report in reports.values())
    if strategy_enabled:
        assert reports[KEY].decisions[1].theoretical_entry is not None
        assert reports[KEY].trades == ()
    else:
        assert reports == {}
    summary = runner.persist(tmp_path / classification.value)
    assert summary["shadow_strategy_enabled"] is strategy_enabled
    assert summary["live_orders_called"] is False


def test_not_ready_missing_stale_and_context_gates_polling(tmp_path: Path) -> None:
    batches, times = _normal_batches()
    for config, expected in (
        (
            _config(tmp_path, classification=ReadinessClassification.NOT_READY_FOR_SHADOW),
            READINESS_NOT_READY,
        ),
        (replace(_config(tmp_path), readiness_report=None), READINESS_REPORT_MISSING),
        (
            _config(
                tmp_path,
                checked_at=datetime(2026, 9, 7, 8, 0, tzinfo=IST),
            ),
            READINESS_STALE,
        ),
        (
            _config(tmp_path, trade_date=date(2026, 9, 8)),
            READINESS_CONTEXT_MISMATCH,
        ),
        (
            replace(
                _config(tmp_path, capital="200000"),
                readiness_report=_config(tmp_path, capital="100000").readiness_report,
            ),
            READINESS_CONTEXT_MISMATCH,
        ),
    ):
        source = _CountingSource(batches)
        runner = ShadowLiveRunner(
            config=config,
            source=source,
            now=iter(times).__next__,
            session_day=TRADE_DATE,
        )
        with pytest.raises(ReadinessGateError, match=expected):
            runner.run()
        assert source.calls == 0
        assert runner.normalized == ()


@pytest.mark.parametrize(
    "code", [CAS_POLICY_UNKNOWN, QUOTE_STALE, FEED_UNAVAILABLE, FEED_GAP_DETECTED]
)
def test_failed_readiness_evidence_cannot_reach_execution(tmp_path: Path, code: str) -> None:
    config = _with_reason(_config(tmp_path), code)
    source = _CountingSource([{}])
    runner = ShadowLiveRunner(
        config=config,
        source=source,
        now=lambda: datetime(2026, 9, 7, 9, 15, 5, tzinfo=IST),
        session_day=TRADE_DATE,
    )

    with pytest.raises(ReadinessGateError, match=READINESS_NOT_READY):
        runner.run()
    assert source.calls == 0
    assert runner.normalized == ()


def test_cas_auxiliary_is_retained_raw_but_excluded_from_strategy(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        cas_eligible=True,
        max_polls=1,
        checked_at=datetime(2026, 9, 7, 15, 16, 5, tzinfo=IST),
    )
    batches = [
        {KEY: _quote(datetime(2026, 9, 7, 15, 16, tzinfo=IST), 102)},
    ]
    times = [
        datetime(2026, 9, 7, 15, 16, 5, tzinfo=IST),
    ]
    runner = _runner(config, batches, times)
    reports = runner.run()
    output = tmp_path / "cas"
    runner.persist(output)

    raw = [json.loads(line) for line in (output / "market_events.jsonl").read_text().splitlines()]
    assert raw[-1]["event"]["is_cas_auxiliary"] is True
    assert reports[KEY].decisions[-1].reason == "cas_auxiliary_excluded_no_trade"
    assert reports[KEY].decisions[-1].theoretical_entry is None
    assert reports[KEY].trades == ()


def test_research_identity_replay_cost_and_credentials_are_deterministic(tmp_path: Path) -> None:
    config = _config(tmp_path)
    batches, times = _normal_batches()
    runner = _runner(config, batches, times)
    reports = runner.run()
    output = tmp_path / "replay"
    summary = runner.persist(output)
    replay = verify_persisted_replay(output, config=config, session_day=TRADE_DATE)

    assert reports[KEY].decisions[1].theoretical_entry is not None
    assert summary["report_fingerprints"][KEY] == replay["fingerprints"][KEY]
    assert replay["matched"][KEY] is True
    scenario = runner._engines[KEY]._cost_scenario
    assert scenario.historical_actual is False
    assert scenario.to_dict()["historical_actual"] is False
    assert all(item.theoretical_exit is None for item in reports[KEY].decisions[:2])
    assert scan_output_for_credentials(output, "synthetic-token-that-must-not-persist") == []
    assert all(item.live_orders_called is False for item in reports[KEY].decisions)
    assert reports[KEY].live_orders_called is False


def test_readiness_report_round_trip_and_mutation_fail_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = config.readiness_report
    assert report is not None
    path = tmp_path / "readiness-report.json"
    path.write_text(report.to_json(), encoding="utf-8")
    loaded = LiveMarketReadinessReport.from_dict(json.loads(path.read_text()))
    assert loaded.to_dict() == report.to_dict()

    mutated = json.loads(path.read_text())
    mutated["context"]["quote_feed_fingerprint"] = "0" * 64
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        LiveMarketReadinessReport.from_dict(json.loads(path.read_text()))


def test_shadow_surface_has_no_broker_order_api_or_credential_persistence() -> None:
    root = Path(__file__).parents[1]
    sources = (
        root / "src/equity_engine/shadow_live_runner.py",
        root / "src/equity_engine/live_market_readiness.py",
        root / "scripts/shadow_live.py",
        root / "src/equity_engine/shadow_execution.py",
    )
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert all(name not in source_text for name in ("place_order", "modify_order", "cancel_order"))
    assert "UPSTOX_ACCESS_TOKEN" in source_text
    assert "live_orders_called" in source_text


def test_monday_probe_to_shadow_rehearsal_persists_healthy_trade_and_replays(
    tmp_path: Path,
) -> None:
    checked_at = datetime(2026, 9, 7, 14, 30, 5, tzinfo=IST)
    probe = _probe_readiness_report(checked_at)
    assert probe.report is not None
    assert probe.report.classification is ReadinessClassification.READY_FOR_RESEARCH_SHADOW
    assert probe.live_orders_called is False

    config = _config(
        tmp_path,
        checked_at=checked_at,
        exit_buffer_minutes=59,
        max_polls=4,
    )
    config = replace(config, readiness_report=probe.report)
    bar_times = (
        datetime(2026, 9, 7, 14, 30, tzinfo=IST),
        datetime(2026, 9, 7, 14, 30, 5, tzinfo=IST),
        datetime(2026, 9, 7, 14, 30, 10, tzinfo=IST),
        datetime(2026, 9, 7, 14, 31, tzinfo=IST),
    )
    received_times = tuple(item + timedelta(seconds=5) for item in bar_times)
    runner = ShadowLiveRunner(
        config=config,
        source=SyntheticQuoteSource(
            [{KEY: _quote(item, 100 + index)} for index, item in enumerate(bar_times)]
        ),
        now=iter(received_times).__next__,
        session_day=TRADE_DATE,
    )

    reports = runner.run()
    output = tmp_path / "monday-rehearsal"
    summary = runner.persist(output)
    replay = verify_persisted_replay(output, config=config, session_day=TRADE_DATE)

    assert len(reports[KEY].trades) == 1
    assert reports[KEY].trades[0].label == "theoretical_never_broker_confirmed"
    assert summary["live_orders_called"] is False
    assert runner.health_report is not None
    assert runner.health_report.status.value == "HEALTHY"
    assert runner.health_report.theoretical_trades_count == 1
    assert json.loads((output / "health-report.json").read_text())["status"] == "HEALTHY"
    assert replay["matched"] == {KEY: True}


def test_degraded_health_stops_new_theoretical_trade_generation(tmp_path: Path) -> None:
    config = _config(tmp_path, max_polls=4)
    batches = [
        {KEY: _quote(datetime(2026, 9, 7, 9, 15, tzinfo=IST), 100)},
        {KEY: _quote(datetime(2026, 9, 7, 9, 20, tzinfo=IST), 101)},
        {KEY: _quote(datetime(2026, 9, 7, 9, 30, tzinfo=IST), 102)},
        {KEY: _quote(datetime(2026, 9, 7, 9, 35, tzinfo=IST), 103)},
    ]
    received = [
        datetime(2026, 9, 7, 9, 15, 5, tzinfo=IST),
        datetime(2026, 9, 7, 9, 20, 5, tzinfo=IST),
        datetime(2026, 9, 7, 9, 30, 5, tzinfo=IST),
        datetime(2026, 9, 7, 9, 35, 5, tzinfo=IST),
    ]
    runner = ShadowLiveRunner(
        config=config,
        source=SyntheticQuoteSource(batches),
        now=iter(received).__next__,
        session_day=TRADE_DATE,
    )

    reports = runner.run()

    assert len(reports[KEY].decisions) == 3
    assert reports[KEY].decisions[-1].reason == "feed_gap_no_trade"
    assert reports[KEY].trades == ()
    assert runner.health_report is not None
    assert runner.health_report.status.value == "DEGRADED_NO_TRADING"
    assert runner.health_report.allow_new_theoretical_trades is False
    assert runner.health_report.live_orders_called is False
