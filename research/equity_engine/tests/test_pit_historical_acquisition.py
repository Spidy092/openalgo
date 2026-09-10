from __future__ import annotations

import hashlib
import json
import runpy
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import date, time
from decimal import Decimal
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from equity_engine.documented_costs import CurrentTermsNSEIntradayCostProvider
from equity_engine.pit_historical_acquisition import (
    AcquisitionRateLimit,
    CorporateActionEvidenceClaim,
    IncompleteUniverseManifestError,
    IncompletePrefilterError,
    PITFormationPolicy,
    PITHistoricalAcquisitionPlan,
    PITResearchBoundary,
    StageAPrefilterResult,
    build_stage_a_plan,
    build_stage_a_prefilter,
    build_stage_b_plan,
    load_pit_universe_source,
    stage_a_batch_candidates,
    write_acquisition_plan,
    write_stage_a_prefilter,
)
from equity_engine.tick_size import FixedTickSizePolicy
from equity_engine.universe import CorporateActionAssessment, ResearchUniverseThresholds


def _source_manifest(tmp_path: Path) -> Path:
    # 2026-09-04 (Friday) is sourced lookback history before the boundary and is
    # ineligible, so the strict trading-session lookback resolves without
    # changing the boundary population.
    dates = (date(2026, 9, 4), date(2026, 9, 7), date(2026, 9, 8))
    eligible_by_date = {date(2026, 9, 4): False, date(2026, 9, 7): True, date(2026, 9, 8): True}
    daily_dir = tmp_path / "daily" / "2026"
    daily_dir.mkdir(parents=True)
    days = []
    for trade_date in dates:
        raw_dir = tmp_path / "raw" / "2026"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"NSE_CM_security_{trade_date.strftime('%d%m%Y')}.csv.gz"
        raw_bytes = f"raw-{trade_date.isoformat()}".encode("utf-8")
        raw_path.write_bytes(raw_bytes)
        raw_hash = hashlib.sha256(raw_bytes).hexdigest()
        raw_path.with_suffix(raw_path.suffix + ".sha256").write_text(
            raw_hash + "\n", encoding="utf-8"
        )
        parquet = daily_dir / f"NSE_CM_universe_{trade_date.isoformat()}.parquet"
        pd.DataFrame(
            [
                {
                    "report_date": trade_date.isoformat(),
                    "instrument_key": "NSE_EQ|INE000000001",
                    "symbol": "OPEN",
                    "isin": "INE000000001",
                    "series": "EQ",
                    "tick_size_rupees": "0.05",
                    "eligible": eligible_by_date[trade_date],
                    "snapshot_sha256": raw_hash,
                    "source_row_number": 2,
                    "source_url": (
                        "https://nsearchives.nseindia.com/web/sites/default/files/"
                        f"NSE_CM_security_{trade_date.strftime('%d%m%Y')}.csv.gz"
                    ),
                }
            ]
        ).to_parquet(parquet, index=False)
        days.append(
            {
                "report_date": trade_date.isoformat(),
                "snapshot_sha256": raw_hash,
                "records": 1,
                "eligible_records": 1 if eligible_by_date[trade_date] else 0,
                "source_url": (
                    "https://nsearchives.nseindia.com/web/sites/default/files/"
                    f"NSE_CM_security_{trade_date.strftime('%d%m%Y')}.csv.gz"
                ),
                "raw_gzip": str(raw_path.relative_to(tmp_path)),
                "universe_parquet": str(parquet.relative_to(tmp_path)),
            }
        )
    manifest = tmp_path / "nse_universe_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "calendar": {
                    "normal_trading_dates": [item.isoformat() for item in dates],
                    "holiday_dates": (),
                    "excluded_special_session_dates": (),
                },
                "summary": {"status": "success", "trading_dates": len(dates)},
                "days": days,
                "failures": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest


def _policy() -> PITFormationPolicy:
    return PITFormationPolicy(
        policy_id="pit-prior-close-v1",
        timezone="Asia/Kolkata",
        decision_time=time(9, 20),
        price_reference_policy="prior_completed_session_close",
        signal_time_policy="signal-after-universe-formation",
        execution_time_policy="execution-after-signal-on-same-session",
    )


def _rate_limit() -> AcquisitionRateLimit:
    return AcquisitionRateLimit(
        policy_id="upstox-evidence-rate-limit-v1",
        minimum_interval_seconds=Decimal("0.20"),
        max_attempts=4,
        backoff_seconds=Decimal("1"),
        source_reference="operator-approved-rate-limit-evidence",
    )


def _thresholds() -> ResearchUniverseThresholds:
    return ResearchUniverseThresholds(
        max_last_price_rupees=Decimal("500"),
        min_median_daily_notional_proxy_rupees=Decimal("0"),
        min_median_daily_volume_shares=Decimal("0"),
        min_observed_trading_days=1,
        min_affordable_quantity=1,
    )


def _plan(tmp_path: Path):
    manifest = _source_manifest(tmp_path)
    return build_stage_a_plan(
        universe_manifest_path=manifest,
        boundary=PITResearchBoundary(start=date(2026, 9, 7), end=date(2026, 9, 8)),
        approved_capital_rupees=Decimal("1000"),
        thresholds=_thresholds(),
        formation_policy=_policy(),
        rate_limit=_rate_limit(),
        universe_rule_version="nse-cm-v15-point-in-time",
        adjustment_policy="raw-unadjusted-block-structural-actions",
        lookback_trading_sessions=1,
        estimated_rows_per_trading_day=1,
        estimated_bytes_per_row=80,
        cost_model_identity="documented-current-terms-explicit-scenario",
    )


def _daily_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-09-07 15:30", tz="Asia/Kolkata"),
            pd.Timestamp("2026-09-08 15:30", tz="Asia/Kolkata"),
        ]
    )
    return pd.DataFrame(
        {
            "open": [99, 109],
            "high": [101, 111],
            "low": [98, 108],
            "close": [100, 110],
            "volume": [100_000, 100_000],
        },
        index=index,
    )


def _claim(key: str, cutoff: date, start: date) -> CorporateActionEvidenceClaim:
    return CorporateActionEvidenceClaim(
        instrument_key=key,
        assessment_as_of=cutoff,
        coverage_start=start,
        coverage_end=cutoff,
        source_fingerprint="e" * 64,
        policy_identity="ca-policy-v1",
        blocking_events=(),
        complete=True,
    )


def _prefilter(
    plan, frame: pd.DataFrame | None = None, *, with_claims: bool = False
) -> StageAPrefilterResult:
    key = "NSE_EQ|INE000000001"
    kwargs: dict[str, object] = {}
    if with_claims:
        kwargs["corporate_action_claims"] = {key: _claim(key, plan.boundary.end, date(2026, 9, 7))}
    return build_stage_a_prefilter(
        stage_a_plan=plan,
        daily_frames={key: frame if frame is not None else _daily_frame()},
        daily_dataset_fingerprints={key: "d" * 64},
        tick_policies={
            key: FixedTickSizePolicy(
                tick_size_rupees=Decimal("0.05"), source="nse-dated-tick-fixture"
            )
        },
        corporate_actions={key: CorporateActionAssessment(complete=True, blocking_events=())},
        minimum_tradable_quantities={key: 1},
        cost_provider=CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
        selection_cutoff=plan.boundary.end,
        **kwargs,  # type: ignore[arg-type]
    )


def test_source_loader_preserves_actual_dates_and_excludes_no_holiday_as_missing(tmp_path: Path):
    manifest = _source_manifest(tmp_path)
    source = load_pit_universe_source(manifest)

    assert source.trading_dates == (date(2026, 9, 4), date(2026, 9, 7), date(2026, 9, 8))
    assert source.eligible_instrument_keys == ("NSE_EQ|INE000000001",)
    assert source.manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_source_loader_rejects_missing_or_unverified_raw_snapshot(tmp_path: Path):
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    raw_path = tmp_path / payload["days"][0]["raw_gzip"]
    raw_path.unlink()

    with pytest.raises(IncompleteUniverseManifestError, match="raw snapshot"):
        load_pit_universe_source(manifest)


def test_stage_a_plan_is_deterministic_and_has_explicit_estimates(tmp_path: Path):
    first = _plan(tmp_path)
    second = build_stage_a_plan(
        universe_manifest_path=Path(first.source.manifest_path),
        boundary=first.boundary,
        approved_capital_rupees=first.approved_capital_rupees,
        thresholds=first.thresholds,
        formation_policy=first.formation_policy,
        rate_limit=first.rate_limit,
        universe_rule_version=first.universe_rule_version,
        adjustment_policy=first.adjustment_policy,
        lookback_trading_sessions=first.lookback_trading_sessions,
        estimated_rows_per_trading_day=first.estimated_rows_per_trading_day,
        estimated_bytes_per_row=first.estimated_bytes_per_row,
        cost_model_identity=first.cost_model_identity,
    )

    assert first.fingerprint == second.fingerprint
    assert first.source.manifest_sha256
    assert first.estimated_requests == 1
    assert first.estimated_rows == 2
    assert first.estimated_storage_bytes == 160
    assert stage_a_batch_candidates(first)[0].instrument_key == "NSE_EQ|INE000000001"
    assert first.as_dict()["live_orders_called"] is False


def test_stage_a_prefilter_uses_only_price_available_before_formation(tmp_path: Path):
    plan = _plan(tmp_path)
    baseline = _prefilter(plan)
    changed_future = _daily_frame()
    future_timestamp = pd.Timestamp("2026-09-08 15:30", tz="Asia/Kolkata")
    changed_future.loc[future_timestamp, "close"] = 490
    changed_future.loc[future_timestamp, "high"] = 490
    future_changed = _prefilter(plan, changed_future)

    assert baseline.decisions[0].reference_price_rupees == Decimal("100")
    assert future_changed.decisions[0].reference_price_rupees == Decimal("100")
    assert future_changed.decisions[0].eligible == baseline.decisions[0].eligible


def test_daily_stage_a_rejects_same_day_close_policy_that_could_look_ahead():
    with pytest.raises(ValueError, match="prior completed-session close"):
        PITFormationPolicy(
            policy_id="unsafe-daily-policy",
            timezone="Asia/Kolkata",
            decision_time=time(9, 20),
            price_reference_policy="close_at_or_before_decision_time",
            signal_time_policy="after-formation",
            execution_time_policy="after-signal",
        )


def test_stage_a_prefilter_is_complete_only_when_all_daily_data_is_present(tmp_path: Path):
    plan = _plan(tmp_path)
    missing = _daily_frame().iloc[:0]
    result = _prefilter(plan, missing)

    assert result.complete is False
    assert result.decisions[0].eligible is False
    with pytest.raises(IncompletePrefilterError, match="Stage-A prefilter is complete"):
        build_stage_b_plan(
            stage_a_plan=plan,
            prefilter=result,
            interval_minutes=5,
            expected_rows_per_trading_day=75,
            estimated_bytes_per_row=80,
            rate_limit=_rate_limit(),
            window_cutoffs=(plan.boundary.end,),
        )


def test_stage_b_is_bound_to_complete_prefilter_and_estimates_minute_storage(tmp_path: Path):
    plan = _plan(tmp_path)
    prefilter = _prefilter(plan, with_claims=True)
    stage_b = build_stage_b_plan(
        stage_a_plan=plan,
        prefilter=prefilter,
        interval_minutes=5,
        expected_rows_per_trading_day=75,
        estimated_bytes_per_row=80,
        rate_limit=_rate_limit(),
        window_cutoffs=(plan.boundary.end,),
    )
    combined = PITHistoricalAcquisitionPlan(stage_a=plan, stage_b=stage_b)

    assert stage_b.candidates[0].instrument_key == "NSE_EQ|INE000000001"
    assert stage_b.estimated_requests == 1
    assert stage_b.estimated_rows == 150
    assert stage_b.estimated_storage_bytes == 12_000
    assert stage_b.prefilter_fingerprint == prefilter.fingerprint
    assert combined.as_dict()["live_orders_called"] is False


def test_stage_b_rejects_prefilter_from_another_stage_a_plan(tmp_path: Path):
    plan = _plan(tmp_path)
    prefilter = _prefilter(plan)
    forged = replace(prefilter, stage_a_plan_fingerprint="0" * 64)

    with pytest.raises(IncompletePrefilterError, match="not bound"):
        build_stage_b_plan(
            stage_a_plan=plan,
            prefilter=forged,
            interval_minutes=5,
            expected_rows_per_trading_day=75,
            estimated_bytes_per_row=80,
            rate_limit=_rate_limit(),
            window_cutoffs=(plan.boundary.end,),
        )


def test_plan_artifact_write_is_atomic_and_contains_no_token(tmp_path: Path):
    plan = _plan(tmp_path)
    path = tmp_path / "plan.json"
    write_acquisition_plan(PITHistoricalAcquisitionPlan(stage_a=plan), path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["stage_a"]["deterministic_fingerprint"] == plan.fingerprint
    assert "UPSTOX_ACCESS_TOKEN" not in path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".plan.json.*.tmp"))


def test_stage_a_prefilter_artifact_is_persisted_with_source_and_decision_fingerprints(
    tmp_path: Path,
):
    plan = _plan(tmp_path)
    prefilter = _prefilter(plan)
    path = tmp_path / "stage-a-prefilter.json"

    write_stage_a_prefilter(prefilter, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["complete"] is True
    assert payload["affordability_candidate_count"] == 1
    assert payload["source_manifest_sha256"] == plan.source.manifest_sha256
    assert payload["stage_a_plan_fingerprint"] == plan.fingerprint
    assert payload["deterministic_fingerprint"] == prefilter.fingerprint
    assert payload["live_orders_called"] is False


def test_dry_run_plan_cli_reads_only_the_manifest_and_reports_stage_a_estimates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = _source_manifest(tmp_path)
    output_path = tmp_path / "cli-plan.json"
    argv = [
        "pit_historical_plan.py",
        "--universe-manifest",
        str(manifest),
        "--start",
        "2026-09-07",
        "--end",
        "2026-09-08",
        "--approved-capital",
        "1000",
        "--max-last-price",
        "500",
        "--min-median-daily-notional",
        "0",
        "--min-median-daily-volume",
        "0",
        "--min-observed-trading-days",
        "1",
        "--min-affordable-quantity",
        "1",
        "--timezone",
        "Asia/Kolkata",
        "--formation-policy-id",
        "pit-policy",
        "--decision-time",
        "09:20",
        "--price-reference-policy",
        "prior_completed_session_close",
        "--signal-time-policy",
        "after-formation",
        "--execution-time-policy",
        "after-signal",
        "--rate-limit-policy-id",
        "rate-policy",
        "--rate-limit-source",
        "operator-evidence",
        "--min-request-interval",
        "0.2",
        "--max-attempts",
        "4",
        "--backoff-seconds",
        "1",
        "--universe-rule-version",
        "test-rule",
        "--adjustment-policy",
        "raw",
        "--lookback-trading-sessions",
        "1",
        "--estimated-rows-per-trading-day",
        "1",
        "--estimated-bytes-per-daily-row",
        "80",
        "--cost-model-identity",
        "test-costs",
        "--output",
        str(output_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    stdout = StringIO()
    with redirect_stdout(stdout), pytest.raises(SystemExit) as exit_info:
        runpy.run_path(
            str(Path(__file__).parents[1] / "scripts" / "pit_historical_plan.py"),
            run_name="__main__",
        )

    assert exit_info.value.code == 0
    report = json.loads(stdout.getvalue())
    assert report["mode"] == "dry-run"
    assert report["trading_dates_found"] == 2
    assert report["unique_pit_instruments"] == 1
    assert report["candidate_count_after_affordability_filter"] is None
    assert output_path.exists()
