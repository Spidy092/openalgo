"""Monday readiness probe V1 tests (synthetic, no network, no orders)."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from equity_engine.experiment import (
    ApprovedCapital,
    CostReconciliationEvidence,
    LiveOrderAttemptError,
    PaperTradingEvidence,
)
from equity_engine.historical_validation import (
    DailyIntradayValidation,
    HistoricalDatasetValidation,
)
from equity_engine.live_market_readiness import ReadinessClassification
from equity_engine.monday_readiness_probe import (
    BLOCKED_TOKEN_MISSING,
    EXIT_BLOCKED_TOKEN,
    EXIT_LIVE_REVIEW,
    EXIT_NOT_READY,
    EXIT_RESEARCH_SHADOW,
    EXIT_SHADOW_INFRA,
    TOKEN_ENV_VAR,
    FeedHealthEvidence,
    MondayProbeConfig,
    ProbeMode,
    assemble_inputs,
    blocked_token_result,
    current_ist_now,
    derive_session_state,
    exit_code_for_result,
    parse_cost_evidence_file,
    parse_historical_validation_file,
    parse_paper_evidence_file,
    result_summary,
    run_dry_run,
    run_live_read_only,
    select_instrument,
    token_present_from_env,
    verify_probe_fingerprint,
    verify_tick,
    write_probe_report,
)
from equity_engine.nse_calendar import CalendarEvidence
from equity_engine.upstox_market_context import QuoteBatchResult
from equity_engine.upstox_readiness import ReadinessCheck, UpstoxReadinessSnapshot

IST = ZoneInfo("Asia/Kolkata")
TRADE_DATE = date(2026, 9, 7)  # Monday, not an NSE CM holiday
NOW_IST = datetime(2026, 9, 7, 10, 0, 30, tzinfo=IST)
KEY = "NSE_EQ|INE002A01018"
SECRET = "secret-token-xyz-123"


def _config(**overrides: object) -> MondayProbeConfig:
    fields: dict[str, object] = {
        "mode": ProbeMode.DRY_RUN,
        "instrument_key": KEY,
        "approved_capital_rupees": Decimal(10000),
        "cost_tolerance_inr": Decimal("0.01"),
        "max_quote_age_seconds": 60.0,
        "exit_buffer_minutes": 15,
        "tick_size_scale_rupees_per_raw_unit": Decimal("0.01"),
        "tick_reference_price_rupees": Decimal(500),
        "pit_complete": False,
        "historical_validation": None,
        "strategy_evidence_present": False,
        "cost_evidence": None,
        "paper_evidence": None,
        "feed": None,
        "kill_switch_engaged": False,
    }
    fields.update(overrides)
    return MondayProbeConfig(**fields)  # type: ignore[arg-type]


def _live_config(**overrides: object) -> MondayProbeConfig:
    return _config(mode=ProbeMode.LIVE_READ_ONLY, **overrides)


def _snapshot(*, static_ip: bool = True) -> UpstoxReadinessSnapshot:
    return UpstoxReadinessSnapshot(
        checks=(
            ReadinessCheck(name="profile_api", passed=True, detail="ok"),
            ReadinessCheck(name="nse_enabled", passed=True, detail="ok"),
            ReadinessCheck(name="intraday_product_enabled", passed=True, detail="ok"),
            ReadinessCheck(name="funds_api", passed=True, detail="ok"),
            ReadinessCheck(name="minimum_test_capital_present", passed=True, detail="ok"),
            ReadinessCheck(
                name="primary_static_ip",
                passed=static_ip,
                detail="registered" if static_ip else "not registered",
            ),
        ),
        available_to_trade=Decimal(15000),
        exchanges=("NSE", "BSE"),
        products=("D", "I"),
        primary_static_ip_configured=static_ip,
        secondary_static_ip_configured=False,
    )


def _quotes(*, timestamp: datetime = NOW_IST) -> QuoteBatchResult:
    return QuoteBatchResult(
        requested_instrument_keys=(KEY,),
        quotes={
            KEY: {
                "instrument_token": KEY,
                "last_price": "100",
                "timestamp": timestamp.isoformat(),
            }
        },
        failures={},
        request_count=1,
    )


def _bod_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
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
    row.update(overrides)
    return row


def _mis_rows() -> tuple[dict[str, object], ...]:
    return ({"instrument_key": KEY},)


def _feed() -> FeedHealthEvidence:
    return FeedHealthEvidence(available=True, gap_detected=False, last_heartbeat_ist=NOW_IST)


def _cost() -> CostReconciliationEvidence:
    return CostReconciliationEvidence(
        artifact_fingerprint="sha256_cost_recon_artifact_001",
        schema_version="upstox-cost-reconciliation/v1",
        cost_model_name="documented",
        orders_checked=10,
        passed_count=10,
        failed_count=0,
        max_reconciliation_error_inr=Decimal("0.005"),
        tolerance_inr=Decimal("0.01"),
        status="PASS",
    )


def _historical() -> HistoricalDatasetValidation:
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
        deterministic_data_fingerprint="f" * 64,
        manifest_fingerprint_reference="f" * 64,
        fingerprint_schema="equity-market-data-v2",
        manifest_reference="synthetic-manifest",
        timezone="Asia/Kolkata",
        calendar_evidence=None,
    )


def _paper() -> PaperTradingEvidence:
    return PaperTradingEvidence(
        artifact_fingerprint="sha256_paper_trading_artifact_001",
        environment="upstox_sandbox_v2",
        session_start=date(2026, 8, 1),
        session_end=date(2026, 8, 29),
        verified_orders_count=45,
        audit_log_fingerprint="sha256_paper_audit_log_001",
        source_reference="broker_sandbox_order_log",
    )


def _live_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "token_present": True,
        "now_ist": NOW_IST,
        "readiness_snapshot": _snapshot(),
        "quotes": _quotes(),
        "bod_rows": (_bod_row(),),
        "mis_rows": _mis_rows(),
        "suspended_rows": (),
    }
    kwargs.update(overrides)
    return kwargs


def _research_evidence() -> dict[str, object]:
    return {
        "pit_complete": True,
        "historical_validation": _historical(),
        "strategy_evidence_present": True,
        "cost_evidence": _cost(),
    }


# DRY_RUN mode.


def test_dry_run_reaches_shadow_infra_with_token() -> None:
    result = run_dry_run(_config(), now_ist=NOW_IST, token_present=True)

    assert result.blocked_code is None
    assert result.report is not None
    assert result.report.classification is ReadinessClassification.READY_FOR_SHADOW_INFRA
    assert result.token_present is True
    assert result.live_orders_called is False
    assert verify_probe_fingerprint(result) is True


def test_dry_run_without_token_fails_closed() -> None:
    result = run_dry_run(_config(), now_ist=NOW_IST, token_present=False)

    assert result.report is not None
    assert result.report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert "TOKEN_MISSING" in result.report.reason_codes


def test_dry_run_rejects_live_config() -> None:
    with pytest.raises(ValueError, match="DRY_RUN"):
        run_dry_run(_live_config(), now_ist=NOW_IST, token_present=True)


def test_live_runner_rejects_dry_run_config() -> None:
    with pytest.raises(ValueError, match="LIVE_READ_ONLY"):
        run_live_read_only(_config(), **_live_kwargs())  # type: ignore[arg-type]


# LIVE_READ_ONLY mode: token gate.


def test_live_blocked_token_missing_touches_nothing() -> None:
    result = run_live_read_only(
        _live_config(),
        token_present=False,
        now_ist=NOW_IST,
        readiness_snapshot=None,
        quotes=None,
        bod_rows=(),
        mis_rows=(),
        suspended_rows=(),
    )

    assert result.blocked_code == BLOCKED_TOKEN_MISSING
    assert result.report is None
    assert result.token_present is False
    assert result.live_orders_called is False
    assert verify_probe_fingerprint(result) is True
    assert exit_code_for_result(result) == EXIT_BLOCKED_TOKEN


def test_blocked_token_result_factory() -> None:
    result = blocked_token_result()

    assert result.blocked_code == BLOCKED_TOKEN_MISSING
    assert result.report is None
    assert verify_probe_fingerprint(result) is True


# LIVE_READ_ONLY mode: evidence assembly.


def test_live_reaches_research_shadow_without_paper() -> None:
    result = run_live_read_only(
        _live_config(**_research_evidence(), feed=_feed()), **_live_kwargs()
    )

    assert result.report is not None
    assert result.report.classification is ReadinessClassification.READY_FOR_RESEARCH_SHADOW
    assert "PAPER_EVIDENCE_MISSING" in result.report.reason_codes
    assert exit_code_for_result(result) == EXIT_RESEARCH_SHADOW


def test_live_full_evidence_reaches_live_review() -> None:
    result = run_live_read_only(
        _live_config(**_research_evidence(), feed=_feed(), paper_evidence=_paper()),
        **_live_kwargs(),
    )

    assert result.report is not None
    assert result.report.classification is ReadinessClassification.READY_FOR_LIVE_ORDER_REVIEW
    assert result.report.reason_codes == ()
    assert exit_code_for_result(result) == EXIT_LIVE_REVIEW


def test_live_first_session_reaches_infra() -> None:
    result = run_live_read_only(
        _live_config(feed=_feed()),
        **_live_kwargs(),
    )

    assert result.report is not None
    assert result.report.classification is ReadinessClassification.READY_FOR_SHADOW_INFRA
    assert exit_code_for_result(result) == EXIT_SHADOW_INFRA


def test_live_stale_quote_fails_closed() -> None:
    stale = datetime(2026, 9, 7, 9, 0, 0, tzinfo=IST)
    result = run_live_read_only(
        _live_config(**_research_evidence(), feed=_feed()),
        **_live_kwargs(quotes=_quotes(timestamp=stale)),
    )

    assert result.report is not None
    assert result.report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert "QUOTE_STALE" in result.report.reason_codes
    assert exit_code_for_result(result) == EXIT_NOT_READY


def test_live_missing_instrument_fails_closed() -> None:
    result = run_live_read_only(
        _live_config(**_research_evidence(), feed=_feed()),
        **_live_kwargs(bod_rows=(_bod_row(instrument_key="NSE_EQ|INE999A01010"),)),
    )

    assert result.report is not None
    assert result.report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert "INSTRUMENT_IDENTITY_INCOMPLETE" in result.report.reason_codes
    assert "CAS_POLICY_UNKNOWN" in result.report.reason_codes


def test_live_suspended_instrument_fails_closed() -> None:
    suspended = dict(_bod_row())
    suspended["instrument_type"] = "EQ"
    result = run_live_read_only(
        _live_config(**_research_evidence(), feed=_feed()),
        **_live_kwargs(suspended_rows=(suspended,)),
    )

    assert result.report is not None
    assert result.report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert "SUSPENSION_ACTIVE" in result.report.reason_codes


def test_live_feed_gap_fails_closed() -> None:
    gap = FeedHealthEvidence(available=True, gap_detected=True, last_heartbeat_ist=NOW_IST)
    result = run_live_read_only(_live_config(**_research_evidence(), feed=gap), **_live_kwargs())

    assert result.report is not None
    assert result.report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert "FEED_GAP_DETECTED" in result.report.reason_codes


def test_live_kill_switch_fails_closed() -> None:
    result = run_live_read_only(
        _live_config(**_research_evidence(), feed=_feed(), kill_switch_engaged=True),
        **_live_kwargs(),
    )

    assert result.report is not None
    assert result.report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert "KILL_SWITCH_ENGAGED" in result.report.reason_codes


# Pure helpers.


def test_assemble_inputs_maps_snapshot_balance() -> None:
    from equity_engine.market_sessions import NSEEquitySessionPolicy

    policy = NSEEquitySessionPolicy(cas_eligible=False, exit_buffer_minutes=15)
    with_balance = assemble_inputs(
        config=_live_config(),
        token_present=True,
        readiness_snapshot=_snapshot(),
        now_ist=NOW_IST,
        trade_date=TRADE_DATE,
        is_trading_day=True,
        session_open=True,
        session_policy=policy,
        calendar_evidence=None,
        quotes=_quotes(),
        feed=_feed(),
        instrument=None,
        tick_verification=None,
    )

    assert with_balance.broker_available_to_trade == Decimal(15000)
    assert with_balance.approved_capital == ApprovedCapital(amount_rupees=Decimal(10000))

    without_snapshot = assemble_inputs(
        config=_live_config(),
        token_present=True,
        readiness_snapshot=None,
        now_ist=NOW_IST,
        trade_date=TRADE_DATE,
        is_trading_day=True,
        session_open=True,
        session_policy=policy,
        calendar_evidence=None,
        quotes=_quotes(),
        feed=_feed(),
        instrument=None,
        tick_verification=None,
    )
    assert without_snapshot.broker_available_to_trade is None


def test_derive_session_state() -> None:
    from equity_engine.market_sessions import NSEEquitySessionPolicy

    calendar = CalendarEvidence(
        trading_dates=(TRADE_DATE,),
        holiday_dates=(),
        excluded_special_session_dates=(),
        source_urls=("synthetic",),
    )
    policy = NSEEquitySessionPolicy(cas_eligible=False, exit_buffer_minutes=15)

    assert derive_session_state(
        calendar=calendar, session_policy=policy, trade_date=TRADE_DATE, now_ist=NOW_IST
    ) == (True, True)
    assert derive_session_state(
        calendar=None, session_policy=policy, trade_date=TRADE_DATE, now_ist=NOW_IST
    ) == (False, False)
    assert derive_session_state(
        calendar=calendar, session_policy=None, trade_date=TRADE_DATE, now_ist=NOW_IST
    ) == (True, False)
    night = datetime(2026, 9, 7, 20, 0, 0, tzinfo=IST)
    assert derive_session_state(
        calendar=calendar, session_policy=policy, trade_date=TRADE_DATE, now_ist=night
    ) == (True, False)


def test_select_instrument_and_tick() -> None:
    instrument = select_instrument(
        bod_rows=(_bod_row(),),
        mis_rows=_mis_rows(),
        suspended_rows=(),
        as_of_date=TRADE_DATE,
        tick_size_scale_rupees_per_raw_unit=Decimal("0.01"),
        instrument_key=KEY,
    )

    assert instrument is not None
    assert instrument.live_tradability_proven is True
    tick = verify_tick(
        instrument=instrument,
        trade_date=TRADE_DATE,
        reference_price_rupees=Decimal(500),
    )
    assert tick is not None and tick.passed is True

    assert (
        select_instrument(
            bod_rows=(),
            mis_rows=(),
            suspended_rows=(),
            as_of_date=TRADE_DATE,
            tick_size_scale_rupees_per_raw_unit=Decimal("0.01"),
            instrument_key=KEY,
        )
        is None
    )
    assert (
        verify_tick(instrument=None, trade_date=TRADE_DATE, reference_price_rupees=Decimal(100))
        is None
    )


def test_config_validation_rejects_hidden_defaults() -> None:
    with pytest.raises(ValueError, match="instrument_key"):
        _config(instrument_key="  ")
    with pytest.raises(ValueError, match="approved_capital"):
        _config(approved_capital_rupees=Decimal(0))
    with pytest.raises(ValueError, match="exit_buffer_minutes"):
        _config(exit_buffer_minutes=60)
    with pytest.raises(ValueError, match="max_quote_age_seconds"):
        _config(max_quote_age_seconds=0.0)
    with pytest.raises(TypeError, match="mode"):
        _config(mode="LIVE_READ_ONLY")  # type: ignore[arg-type]


def test_current_ist_now_is_aware() -> None:
    now = current_ist_now()

    assert now.tzinfo is not None


# Token presence: boolean only, never the value.


def test_token_presence_is_boolean_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(TOKEN_ENV_VAR, SECRET)
    assert token_present_from_env() is True

    result = run_dry_run(_config(), now_ist=NOW_IST, token_present=token_present_from_env())
    assert SECRET not in result.to_json()

    path = tmp_path / "probe.json"
    write_probe_report(path, result)
    assert SECRET not in path.read_text(encoding="utf-8")

    summary = result_summary(result)
    assert summary["token_present"] is True
    assert SECRET not in json.dumps(summary)

    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    assert token_present_from_env() is False


# Fingerprint + persistence.


def test_fingerprint_is_deterministic() -> None:
    first = run_dry_run(_config(), now_ist=NOW_IST, token_present=True)
    second = run_dry_run(_config(), now_ist=NOW_IST, token_present=True)

    assert first.artifact_fingerprint == second.artifact_fingerprint
    assert verify_probe_fingerprint(first) is True

    different = run_dry_run(_config(), now_ist=NOW_IST, token_present=False)
    assert different.artifact_fingerprint != first.artifact_fingerprint


def test_persisted_report_round_trips(tmp_path: Path) -> None:
    result = run_live_read_only(
        _live_config(**_research_evidence(), feed=_feed(), paper_evidence=_paper()),
        **_live_kwargs(),
    )
    path = tmp_path / "nested" / "readiness.json"
    write_probe_report(path, result)

    payload = json.loads(path.read_text(encoding="utf-8"))
    reference = payload.pop("artifact_fingerprint")
    from equity_engine.current_market_discovery import fingerprint_payload

    assert fingerprint_payload(payload) == reference
    assert payload["live_orders_called"] is False
    assert payload["report"]["live_orders_called"] is False


def test_write_refuses_unverifiable_fingerprint(tmp_path: Path) -> None:
    result = run_dry_run(_config(), now_ist=NOW_IST, token_present=True)
    tampered = replace(result, artifact_fingerprint="0" * 64)

    with pytest.raises(ValueError, match="does not verify"):
        write_probe_report(tmp_path / "probe.json", tampered)


def test_blocked_result_cannot_carry_report() -> None:
    result = run_dry_run(_config(), now_ist=NOW_IST, token_present=True)

    with pytest.raises(ValueError, match="blocked probe"):
        replace(result, blocked_code=BLOCKED_TOKEN_MISSING)


def test_live_orders_guard() -> None:
    result = run_dry_run(_config(), now_ist=NOW_IST, token_present=True)

    with pytest.raises(LiveOrderAttemptError, match="strictly forbidden"):
        replace(result, live_orders_called=True)


# Strict evidence-file parsers.


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_parse_cost_evidence_file(tmp_path: Path) -> None:
    valid = {
        "artifact_fingerprint": "a" * 64,
        "schema_version": "upstox-cost-reconciliation/v1",
        "cost_model_name": "documented",
        "orders_checked": 10,
        "passed_count": 10,
        "failed_count": 0,
        "max_reconciliation_error_inr": "0.005",
        "tolerance_inr": "0.01",
        "status": "PASS",
    }
    parsed = parse_cost_evidence_file(_write(tmp_path / "cost.json", valid))

    assert parsed is not None
    assert parsed.status == "PASS"

    assert parse_cost_evidence_file(_write(tmp_path / "bad.json", {"status": "PASS"})) is None
    assert (
        parse_cost_evidence_file(_write(tmp_path / "fail.json", {**valid, "status": "FAIL"}))
        is None
    )
    assert parse_cost_evidence_file(tmp_path / "missing.json") is None


def test_parse_paper_evidence_file(tmp_path: Path) -> None:
    valid = {
        "artifact_fingerprint": "b" * 64,
        "environment": "upstox_sandbox_v2",
        "session_start": "2026-08-01",
        "session_end": "2026-08-29",
        "verified_orders_count": 45,
        "audit_log_fingerprint": "c" * 64,
        "source_reference": "broker_sandbox_order_log",
    }
    parsed = parse_paper_evidence_file(_write(tmp_path / "paper.json", valid))

    assert parsed is not None
    assert parsed.verified_orders_count == 45

    assert (
        parse_paper_evidence_file(
            _write(tmp_path / "bad.json", {**valid, "session_end": "not-a-date"})
        )
        is None
    )
    assert parse_paper_evidence_file(tmp_path / "missing.json") is None


def test_parse_historical_validation_file_round_trip(tmp_path: Path) -> None:
    import json as _json

    original = _historical()
    parsed = parse_historical_validation_file(
        _write(tmp_path / "hist.json", _json.loads(_json.dumps(original.as_dict())))
    )

    assert parsed is not None
    assert parsed.passed is True
    assert parsed.deterministic_data_fingerprint == original.deterministic_data_fingerprint

    assert parse_historical_validation_file(_write(tmp_path / "bad.json", {"rows": 1})) is None
    assert parse_historical_validation_file(tmp_path / "missing.json") is None


# Monday CLI end to end (no network paths only).


def _load_cli() -> object:
    script = Path(__file__).parents[1] / "scripts" / "monday_readiness_probe.py"
    spec = importlib.util.spec_from_file_location("monday_readiness_probe_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cli_args(tmp_path: Path, mode: str) -> list[str]:
    return [
        "--mode",
        mode,
        "--instrument-key",
        KEY,
        "--approved-capital",
        "10000",
        "--cost-tolerance",
        "0.01",
        "--max-quote-age-seconds",
        "60",
        "--exit-buffer-minutes",
        "15",
        "--tick-size-scale",
        "0.01",
        "--tick-reference-price",
        "100",
        "--output",
        str(tmp_path / "probe.json"),
    ]


def test_cli_dry_run_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cli = _load_cli()
    monkeypatch.setattr(cli, "current_ist_now", lambda: NOW_IST)
    monkeypatch.setenv(TOKEN_ENV_VAR, SECRET)

    code = cli.main(_cli_args(tmp_path, "dry-run"))

    assert code == EXIT_SHADOW_INFRA
    payload = json.loads((tmp_path / "probe.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "DRY_RUN"
    assert payload["report"]["classification"] == "READY_FOR_SHADOW_INFRA"
    assert payload["token_present"] is True
    assert SECRET not in (tmp_path / "probe.json").read_text(encoding="utf-8")


def test_cli_dry_run_without_token_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = _load_cli()
    monkeypatch.setattr(cli, "current_ist_now", lambda: NOW_IST)
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

    code = cli.main(_cli_args(tmp_path, "dry-run"))

    assert code == EXIT_NOT_READY
    payload = json.loads((tmp_path / "probe.json").read_text(encoding="utf-8"))
    assert payload["report"]["classification"] == "NOT_READY_FOR_SHADOW"
    assert "TOKEN_MISSING" in payload["report"]["reason_codes"]


def test_cli_live_blocked_without_token_no_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = _load_cli()
    monkeypatch.setattr(cli, "current_ist_now", lambda: NOW_IST)
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

    code = cli.main(_cli_args(tmp_path, "live-read-only"))

    assert code == EXIT_BLOCKED_TOKEN
    payload = json.loads((tmp_path / "probe.json").read_text(encoding="utf-8"))
    assert payload["blocked_code"] == BLOCKED_TOKEN_MISSING
    assert payload["report"] is None
    assert payload["token_present"] is False


def test_probe_modules_have_no_order_capability() -> None:
    src = Path(__file__).parents[1] / "src" / "equity_engine"
    text = (src / "monday_readiness_probe.py").read_text(encoding="utf-8")
    lowered = text.lower()
    # The TOKEN_ENV_VAR constant legitimately names the env var; the value
    # itself must never be wired anywhere, so forbid kwarg-style handoff.
    for forbidden in (
        "place_order",
        "modify_order",
        "cancel_order",
        "placeorder",
        "cancelorder",
        "access_token=",
        "bearer",
        "authorization",
        "api_key",
        "apikey",
    ):
        assert forbidden not in lowered, f"{forbidden!r} in monday_readiness_probe.py"
    assert "import httpx" not in text
    assert "token_present_from_env" in text  # presence is derived, value never kept
    cli_text = (src.parents[1] / "scripts" / "monday_readiness_probe.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "place_order",
        "modify_order",
        "cancel_order",
        "placeorder",
        "cancelorder",
    ):
        assert forbidden not in cli_text.lower()
