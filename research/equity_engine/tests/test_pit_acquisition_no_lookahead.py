"""P0 no-lookahead / no-survivorship proofs for staged historical acquisition.

Proves:

- A daily candle for selection date T timestamped ``T 00:00 Asia/Kolkata`` with
  an absurd future close cannot change the Stage-A eligibility result.
- A stock eligible in an early window but ineligible at the final boundary
  remains in the Stage-B acquisition union for that early window.
- Prior-session resolution uses sourced normal trading sessions for Monday,
  exchange holidays, new listings, eligibility gaps, delistings, and symbol
  changes. No naive ``first_date - 1 calendar day`` inference.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, time
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from equity_engine.documented_costs import CurrentTermsNSEIntradayCostProvider
from equity_engine.pit_historical_acquisition import (
    AcquisitionRateLimit,
    IncompletePrefilterError,
    PITAcquisitionError,
    PITFormationPolicy,
    PITResearchBoundary,
    build_stage_a_plan,
    build_stage_a_prefilter,
    build_stage_a_prefilter_timeline,
    build_stage_b_plan,
    build_stage_b_plan_from_timeline,
    prior_completed_trading_session,
)
from equity_engine.tick_size import FixedTickSizePolicy
from equity_engine.universe import CorporateActionAssessment, ResearchUniverseThresholds

KEY_A = "NSE_EQ|INE000000001"
KEY_B = "NSE_EQ|INE000000002"


def _write_manifest(
    tmp_path: Path,
    dates: tuple[date, ...],
    rows_by_date: dict[date, list[dict[str, object]]],
) -> Path:
    daily_dir = tmp_path / "daily" / "2026"
    daily_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = tmp_path / "raw" / "2026"
    raw_dir.mkdir(parents=True, exist_ok=True)
    days = []
    for trade_date in dates:
        raw_path = raw_dir / f"NSE_CM_security_{trade_date.strftime('%d%m%Y')}.csv.gz"
        raw_bytes = f"raw-{trade_date.isoformat()}".encode()
        raw_path.write_bytes(raw_bytes)
        raw_hash = hashlib.sha256(raw_bytes).hexdigest()
        raw_path.with_suffix(raw_path.suffix + ".sha256").write_text(raw_hash + "\n")
        parquet = daily_dir / f"NSE_CM_universe_{trade_date.isoformat()}.parquet"
        pd.DataFrame(rows_by_date[trade_date]).to_parquet(parquet, index=False)
        source_url = (
            "https://nsearchives.nseindia.com/web/sites/default/files/"
            f"NSE_CM_security_{trade_date.strftime('%d%m%Y')}.csv.gz"
        )
        # Rewrite rows with canonical source_url/snapshot so loader verification passes.
        frame = pd.read_parquet(parquet)
        frame["source_url"] = source_url
        frame["snapshot_sha256"] = raw_hash
        frame.to_parquet(parquet, index=False)
        days.append(
            {
                "report_date": trade_date.isoformat(),
                "snapshot_sha256": raw_hash,
                "records": len(rows_by_date[trade_date]),
                "eligible_records": sum(1 for row in rows_by_date[trade_date] if row["eligible"]),
                "source_url": source_url,
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
        )
    )
    return manifest


def _row(
    trade_date: date,
    key: str,
    symbol: str,
    *,
    eligible: bool = True,
    isin: str | None = None,
) -> dict[str, object]:
    return {
        "report_date": trade_date.isoformat(),
        "instrument_key": key,
        "symbol": symbol,
        "isin": isin or key.split("|")[1],
        "series": "EQ",
        "tick_size_rupees": "0.05",
        "eligible": eligible,
        "snapshot_sha256": "placeholder",
        "source_row_number": 2,
        "source_url": "placeholder",
    }


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


def _midnight_frame(closes_by_date: dict[date, Decimal]) -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [pd.Timestamp(d.isoformat() + " 00:00", tz="Asia/Kolkata") for d in sorted(closes_by_date)]
    )
    closes = [closes_by_date[d] for d in sorted(closes_by_date)]
    return pd.DataFrame(
        {
            "open": [float(c) - 1 for c in closes],
            "high": [float(c) + 1 for c in closes],
            "low": [float(c) - 2 for c in closes],
            "close": [float(c) for c in closes],
            "volume": [100_000] * len(closes),
        },
        index=index,
    )


def _prefilter_kwargs(plan, frame):  # type: ignore[no-untyped-def]
    key = KEY_A
    return {
        "stage_a_plan": plan,
        "daily_frames": {key: frame},
        "daily_dataset_fingerprints": {key: "d" * 64},
        "tick_policies": {
            key: FixedTickSizePolicy(tick_size_rupees=Decimal("0.05"), source="nse-fixture")
        },
        "corporate_actions": {key: CorporateActionAssessment(complete=True, blocking_events=())},
        "minimum_tradable_quantities": {key: 1},
        "cost_provider": CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
        "selection_cutoff": plan.boundary.end,
    }


def test_midnight_t_candle_with_absurd_close_cannot_change_result(tmp_path: Path) -> None:
    sep7, sep8 = date(2026, 9, 7), date(2026, 9, 8)
    manifest = _write_manifest(
        tmp_path,
        (sep7, sep8),
        {
            sep7: [_row(sep7, KEY_A, "OPEN")],
            sep8: [_row(sep8, KEY_A, "OPEN")],
        },
    )
    plan = build_stage_a_plan(
        universe_manifest_path=manifest,
        boundary=PITResearchBoundary(start=sep7, end=sep8),
        approved_capital_rupees=Decimal("1000"),
        thresholds=_thresholds(),
        formation_policy=_policy(),
        rate_limit=_rate_limit(),
        universe_rule_version="nse-cm-v15-point-in-time",
        adjustment_policy="raw-unadjusted-block-structural-actions",
        lookback_calendar_days=1,
        estimated_rows_per_trading_day=1,
        estimated_bytes_per_row=80,
        cost_model_identity="documented-current-terms-explicit-scenario",
    )
    without_t = _midnight_frame({sep7: Decimal("100")})
    with_t = _midnight_frame({sep7: Decimal("100"), sep8: Decimal("999999")})

    base = build_stage_a_prefilter(**_prefilter_kwargs(plan, without_t))
    leaked = build_stage_a_prefilter(**_prefilter_kwargs(plan, with_t))

    assert base.decisions[0].reference_price_rupees == Decimal("100")
    assert leaked.decisions[0].reference_price_rupees == Decimal("100")
    assert leaked.decisions[0].prior_completed_session == sep7
    assert leaked.decisions[0].eligible == base.decisions[0].eligible
    assert leaked.decisions[0].affordable_quantity == base.decisions[0].affordable_quantity
    assert [d.as_dict() for d in leaked.decisions] == [d.as_dict() for d in base.decisions]


def test_monday_uses_friday_prior_session(tmp_path: Path) -> None:
    fri, mon = date(2026, 9, 4), date(2026, 9, 7)
    manifest = _write_manifest(
        tmp_path,
        (fri, mon),
        {fri: [_row(fri, KEY_A, "OPEN")], mon: [_row(mon, KEY_A, "OPEN")]},
    )
    plan = build_stage_a_plan(
        universe_manifest_path=manifest,
        boundary=PITResearchBoundary(start=fri, end=mon),
        approved_capital_rupees=Decimal("1000"),
        thresholds=_thresholds(),
        formation_policy=_policy(),
        rate_limit=_rate_limit(),
        universe_rule_version="nse-cm-v15-point-in-time",
        adjustment_policy="raw-unadjusted-block-structural-actions",
        lookback_calendar_days=1,
        estimated_rows_per_trading_day=1,
        estimated_bytes_per_row=80,
        cost_model_identity="documented-current-terms-explicit-scenario",
    )
    # Lookback must reach the sourced Friday session, never Sunday.
    assert plan.candidates[0].start == fri
    frame = _midnight_frame({fri: Decimal("100"), mon: Decimal("999999")})
    result = build_stage_a_prefilter(
        **{**_prefilter_kwargs(plan, frame), "selection_cutoff": mon},
    )
    assert result.decisions[0].prior_completed_session == fri
    assert result.decisions[0].reference_price_rupees == Decimal("100")


def test_holiday_following_session_uses_previous_trading_session(tmp_path: Path) -> None:
    tue, thu = date(2026, 9, 8), date(2026, 9, 10)
    # Wednesday 2026-09-09 is absent as an exchange holiday in this sourced calendar.
    assert prior_completed_trading_session(thu, (tue, thu)) == tue
    manifest = _write_manifest(
        tmp_path,
        (tue, thu),
        {tue: [_row(tue, KEY_A, "OPEN")], thu: [_row(thu, KEY_A, "OPEN")]},
    )
    plan = build_stage_a_plan(
        universe_manifest_path=manifest,
        boundary=PITResearchBoundary(start=tue, end=thu),
        approved_capital_rupees=Decimal("1000"),
        thresholds=_thresholds(),
        formation_policy=_policy(),
        rate_limit=_rate_limit(),
        universe_rule_version="nse-cm-v15-point-in-time",
        adjustment_policy="raw-unadjusted-block-structural-actions",
        lookback_calendar_days=1,
        estimated_rows_per_trading_day=1,
        estimated_bytes_per_row=80,
        cost_model_identity="documented-current-terms-explicit-scenario",
    )
    frame = _midnight_frame({tue: Decimal("100"), thu: Decimal("999999")})
    result = build_stage_a_prefilter(
        **{**_prefilter_kwargs(plan, frame), "selection_cutoff": thu},
    )
    assert result.decisions[0].prior_completed_session == tue
    assert result.decisions[0].reference_price_rupees == Decimal("100")


def test_new_listing_with_no_prior_session_fails_closed(tmp_path: Path) -> None:
    mon, tue = date(2026, 9, 7), date(2026, 9, 8)
    manifest = _write_manifest(
        tmp_path,
        (mon, tue),
        {
            mon: [_row(mon, KEY_A, "OPEN")],
            tue: [_row(tue, KEY_A, "OPEN"), _row(tue, KEY_B, "NEWLIST")],
        },
    )
    plan = build_stage_a_plan(
        universe_manifest_path=manifest,
        boundary=PITResearchBoundary(start=mon, end=tue),
        approved_capital_rupees=Decimal("1000"),
        thresholds=_thresholds(),
        formation_policy=_policy(),
        rate_limit=_rate_limit(),
        universe_rule_version="nse-cm-v15-point-in-time",
        adjustment_policy="raw-unadjusted-block-structural-actions",
        lookback_calendar_days=1,
        estimated_rows_per_trading_day=1,
        estimated_bytes_per_row=80,
        cost_model_identity="documented-current-terms-explicit-scenario",
    )
    # KEY_B is eligible only on the cutoff itself, so no prior eligible session exists.
    assert [d for d in plan.candidates if d.instrument_key == KEY_B][0].eligible_dates == (tue,)
    frames = {
        KEY_A: _midnight_frame({mon: Decimal("100"), tue: Decimal("101")}),
        KEY_B: _midnight_frame({tue: Decimal("999999")}),
    }
    result = build_stage_a_prefilter(
        stage_a_plan=plan,
        daily_frames=frames,
        daily_dataset_fingerprints={KEY_A: "a" * 64, KEY_B: "b" * 64},
        tick_policies={
            KEY_A: FixedTickSizePolicy(tick_size_rupees=Decimal("0.05"), source="s"),
            KEY_B: FixedTickSizePolicy(tick_size_rupees=Decimal("0.05"), source="s"),
        },
        corporate_actions={
            KEY_A: CorporateActionAssessment(complete=True, blocking_events=()),
            KEY_B: CorporateActionAssessment(complete=True, blocking_events=()),
        },
        minimum_tradable_quantities={KEY_A: 1, KEY_B: 1},
        cost_provider=CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
        selection_cutoff=tue,
    )
    by_key = {d.instrument_key: d for d in result.decisions}
    assert by_key[KEY_B].eligible is False
    assert by_key[KEY_B].reference_price_rupees is None
    # Legitimate PIT ineligibility (not yet listed) must not block other names.
    assert result.complete is True


def test_eligibility_gap_prior_session_must_be_eligible(tmp_path: Path) -> None:
    mon, tue, wed = date(2026, 9, 7), date(2026, 9, 8), date(2026, 9, 9)
    manifest = _write_manifest(
        tmp_path,
        (mon, tue, wed),
        {
            mon: [_row(mon, KEY_A, "OPEN")],
            tue: [_row(tue, KEY_A, "OPEN", eligible=False)],
            wed: [_row(wed, KEY_A, "OPEN")],
        },
    )
    plan = build_stage_a_plan(
        universe_manifest_path=manifest,
        boundary=PITResearchBoundary(start=mon, end=wed),
        approved_capital_rupees=Decimal("1000"),
        thresholds=_thresholds(),
        formation_policy=_policy(),
        rate_limit=_rate_limit(),
        universe_rule_version="nse-cm-v15-point-in-time",
        adjustment_policy="raw-unadjusted-block-structural-actions",
        lookback_calendar_days=1,
        estimated_rows_per_trading_day=1,
        estimated_bytes_per_row=80,
        cost_model_identity="documented-current-terms-explicit-scenario",
    )
    frames = {
        KEY_A: _midnight_frame({mon: Decimal("100"), tue: Decimal("101"), wed: Decimal("102")})
    }
    kwargs = {
        "stage_a_plan": plan,
        "daily_frames": frames,
        "daily_dataset_fingerprints": {KEY_A: "a" * 64},
        "tick_policies": {KEY_A: FixedTickSizePolicy(tick_size_rupees=Decimal("0.05"), source="s")},
        "corporate_actions": {KEY_A: CorporateActionAssessment(complete=True, blocking_events=())},
        "minimum_tradable_quantities": {KEY_A: 1},
        "cost_provider": CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
    }
    # Wednesday's calendar prior is Tuesday, which is ineligible, so Wednesday fails closed.
    wed_result = build_stage_a_prefilter(**kwargs, selection_cutoff=wed)  # type: ignore[arg-type]
    assert wed_result.decisions[0].eligible is False
    assert wed_result.decisions[0].prior_completed_session == tue
    # Tuesday's prior is Monday (eligible), so the Tuesday window remains decidable.
    tue_result = build_stage_a_prefilter(**kwargs, selection_cutoff=tue)  # type: ignore[arg-type]
    assert tue_result.decisions[0].prior_completed_session == mon
    assert tue_result.decisions[0].reference_price_rupees == Decimal("100")


def test_delisted_stock_remains_in_early_window_union(tmp_path: Path) -> None:
    d1, d2, d3, d4 = (date(2026, 9, 7), date(2026, 9, 8), date(2026, 9, 9), date(2026, 9, 10))
    manifest = _write_manifest(
        tmp_path,
        (d1, d2, d3, d4),
        {
            d1: [_row(d1, KEY_A, "OPEN"), _row(d1, KEY_B, "OLD")],
            d2: [_row(d2, KEY_A, "OPEN"), _row(d2, KEY_B, "OLD")],
            d3: [_row(d3, KEY_A, "OPEN"), _row(d3, KEY_B, "OLD", eligible=False)],
            d4: [_row(d4, KEY_A, "OPEN"), _row(d4, KEY_B, "OLD", eligible=False)],
        },
    )
    plan = build_stage_a_plan(
        universe_manifest_path=manifest,
        boundary=PITResearchBoundary(start=d1, end=d4),
        approved_capital_rupees=Decimal("1000"),
        thresholds=_thresholds(),
        formation_policy=_policy(),
        rate_limit=_rate_limit(),
        universe_rule_version="nse-cm-v15-point-in-time",
        adjustment_policy="raw-unadjusted-block-structural-actions",
        lookback_calendar_days=1,
        estimated_rows_per_trading_day=1,
        estimated_bytes_per_row=80,
        cost_model_identity="documented-current-terms-explicit-scenario",
    )
    frames = {
        KEY_A: _midnight_frame(
            {d1: Decimal("100"), d2: Decimal("101"), d3: Decimal("102"), d4: Decimal("103")}
        ),
        KEY_B: _midnight_frame(
            {d1: Decimal("100"), d2: Decimal("101"), d3: Decimal("102"), d4: Decimal("103")}
        ),
    }
    base_kwargs = {
        "stage_a_plan": plan,
        "daily_frames": frames,
        "daily_dataset_fingerprints": {KEY_A: "a" * 64, KEY_B: "b" * 64},
        "tick_policies": {
            KEY_A: FixedTickSizePolicy(tick_size_rupees=Decimal("0.05"), source="s"),
            KEY_B: FixedTickSizePolicy(tick_size_rupees=Decimal("0.05"), source="s"),
        },
        "corporate_actions": {
            KEY_A: CorporateActionAssessment(complete=True, blocking_events=()),
            KEY_B: CorporateActionAssessment(complete=True, blocking_events=()),
        },
        "minimum_tradable_quantities": {KEY_A: 1, KEY_B: 1},
        "cost_provider": CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
    }
    early = build_stage_a_prefilter(**base_kwargs, selection_cutoff=d2)  # type: ignore[arg-type]
    final = build_stage_a_prefilter(**base_kwargs, selection_cutoff=d4)  # type: ignore[arg-type]
    assert {d.instrument_key: d.eligible for d in early.decisions} == {KEY_A: True, KEY_B: True}
    final_by_key = {d.instrument_key: d for d in final.decisions}
    assert final_by_key[KEY_A].eligible is True
    assert final_by_key[KEY_B].eligible is False

    timeline = build_stage_a_prefilter_timeline(**base_kwargs, selection_cutoffs=(d2, d4))  # type: ignore[arg-type]
    assert [r.selection_cutoff for r in timeline] == [d2, d4]
    union = build_stage_b_plan_from_timeline(
        stage_a_plan=plan,
        prefilters=timeline,
        interval_minutes=5,
        expected_rows_per_trading_day=75,
        estimated_bytes_per_row=80,
        rate_limit=_rate_limit(),
    )
    union_keys = {c.instrument_key for c in union.candidates}
    assert KEY_B in union_keys
    assert union.timeline_cutoffs == (d2, d4)
    assert union.timeline_prefilter_fingerprints == tuple(r.fingerprint for r in timeline)
    by_candidate = {c.instrument_key: c for c in union.candidates}
    assert (by_candidate[KEY_B].start, by_candidate[KEY_B].end) == (d1, d2)

    # A final-day-only plan would drop the delisted name entirely.
    final_only = build_stage_b_plan(
        stage_a_plan=plan,
        prefilter=final,
        interval_minutes=5,
        expected_rows_per_trading_day=75,
        estimated_bytes_per_row=80,
        rate_limit=_rate_limit(),
    )
    assert {c.instrument_key for c in final_only.candidates} == {KEY_A}


def test_symbol_change_uses_pit_symbol_without_lookahead(tmp_path: Path) -> None:
    d1, d2 = date(2026, 9, 7), date(2026, 9, 8)
    manifest = _write_manifest(
        tmp_path,
        (d1, d2),
        {d1: [_row(d1, KEY_A, "OLDNAME")], d2: [_row(d2, KEY_A, "NEWNAME")]},
    )
    plan = build_stage_a_plan(
        universe_manifest_path=manifest,
        boundary=PITResearchBoundary(start=d1, end=d2),
        approved_capital_rupees=Decimal("1000"),
        thresholds=_thresholds(),
        formation_policy=_policy(),
        rate_limit=_rate_limit(),
        universe_rule_version="nse-cm-v15-point-in-time",
        adjustment_policy="raw-unadjusted-block-structural-actions",
        lookback_calendar_days=1,
        estimated_rows_per_trading_day=1,
        estimated_bytes_per_row=80,
        cost_model_identity="documented-current-terms-explicit-scenario",
    )
    assert plan.candidates[0].symbol_by_date == ((d1, "OLDNAME"), (d2, "NEWNAME"))
    frame = _midnight_frame({d1: Decimal("100"), d2: Decimal("999999")})
    result = build_stage_a_prefilter(**_prefilter_kwargs(plan, frame))
    assert result.decisions[0].symbol == "OLDNAME"
    assert result.decisions[0].reference_price_rupees == Decimal("100")
    assert result.decisions[0].prior_completed_session == d1


def test_prior_session_helper_rejects_missing_session() -> None:
    with pytest.raises(PITAcquisitionError, match="no prior completed"):
        prior_completed_trading_session(date(2026, 9, 7), (date(2026, 9, 7),))
    assert prior_completed_trading_session(
        date(2026, 9, 8), (date(2026, 9, 7), date(2026, 9, 8))
    ) == date(2026, 9, 7)


def test_timeline_cutoffs_must_be_sorted_inside_boundary(tmp_path: Path) -> None:
    sep7, sep8 = date(2026, 9, 7), date(2026, 9, 8)
    manifest = _write_manifest(
        tmp_path,
        (sep7, sep8),
        {sep7: [_row(sep7, KEY_A, "OPEN")], sep8: [_row(sep8, KEY_A, "OPEN")]},
    )
    plan = build_stage_a_plan(
        universe_manifest_path=manifest,
        boundary=PITResearchBoundary(start=sep7, end=sep8),
        approved_capital_rupees=Decimal("1000"),
        thresholds=_thresholds(),
        formation_policy=_policy(),
        rate_limit=_rate_limit(),
        universe_rule_version="nse-cm-v15-point-in-time",
        adjustment_policy="raw-unadjusted-block-structural-actions",
        lookback_calendar_days=1,
        estimated_rows_per_trading_day=1,
        estimated_bytes_per_row=80,
        cost_model_identity="documented-current-terms-explicit-scenario",
    )
    frame = _midnight_frame({sep7: Decimal("100"), sep8: Decimal("101")})
    kwargs = {
        "stage_a_plan": plan,
        "daily_frames": {KEY_A: frame},
        "daily_dataset_fingerprints": {KEY_A: "a" * 64},
        "tick_policies": {KEY_A: FixedTickSizePolicy(tick_size_rupees=Decimal("0.05"), source="s")},
        "corporate_actions": {KEY_A: CorporateActionAssessment(complete=True, blocking_events=())},
        "minimum_tradable_quantities": {KEY_A: 1},
        "cost_provider": CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
    }
    with pytest.raises(ValueError, match="sorted unique"):
        build_stage_a_prefilter_timeline(**kwargs, selection_cutoffs=(sep8, sep7))  # type: ignore[arg-type]
    with pytest.raises(IncompletePrefilterError, match="at least one prefilter"):
        build_stage_b_plan_from_timeline(
            stage_a_plan=plan,
            prefilters=(),
            interval_minutes=5,
            expected_rows_per_trading_day=75,
            estimated_bytes_per_row=80,
            rate_limit=_rate_limit(),
        )
