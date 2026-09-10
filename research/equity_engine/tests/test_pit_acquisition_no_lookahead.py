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
from dataclasses import replace
from datetime import date, time, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from equity_engine.documented_costs import CurrentTermsNSEIntradayCostProvider
from equity_engine.pit_historical_acquisition import (
    RAW_ACQUISITION_ONLY,
    AcquisitionRateLimit,
    CorporateActionEvidenceClaim,
    IncompletePrefilterError,
    PITAcquisitionError,
    PITFormationPolicy,
    PITResearchBoundary,
    _filtered_bars_fingerprint,
    build_stage_a_plan,
    build_stage_a_prefilter,
    build_stage_a_prefilter_timeline,
    build_stage_b_plan,
    build_stage_b_plan_from_timeline,
    canonical_eligible_intervals,
    consume_research_bars,
    prior_completed_trading_session,
    stage_b_eligibility_mask_fingerprint,
    symbol_ranges_for,
)
from equity_engine.tick_size import FixedTickSizePolicy
from equity_engine.universe import CorporateActionAssessment, ResearchUniverseThresholds

KEY_A = "NSE_EQ|INE000000001"
KEY_B = "NSE_EQ|INE000000002"


def _prior_trading_day(day: date) -> date:
    current = day - timedelta(days=1)
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return current


def _write_manifest(
    tmp_path: Path,
    dates: tuple[date, ...],
    rows_by_date: dict[date, list[dict[str, object]]],
    *,
    prepend_history: bool = True,
) -> Path:
    if prepend_history:
        # Prepend one sourced ineligible session so the strict trading-session
        # lookback resolves from evidence instead of failing on fixtures.
        seen: dict[str, str] = {}
        for rows in rows_by_date.values():
            for row in rows:
                seen.setdefault(str(row["instrument_key"]), str(row["symbol"]))
        prior = _prior_trading_day(dates[0])
        rows_by_date = {
            prior: [
                _row(prior, key, symbol, eligible=False) for key, symbol in sorted(seen.items())
            ],
            **rows_by_date,
        }
        dates = (prior, *dates)
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


def _ca_claims(
    keys: tuple[str, ...],
    *,
    as_of: date,
    coverage_start: date,
    coverage_end: date,
    fingerprint: str = "e" * 64,
) -> dict[str, CorporateActionEvidenceClaim]:
    return {
        key: CorporateActionEvidenceClaim(
            instrument_key=key,
            assessment_as_of=as_of,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            source_fingerprint=fingerprint,
            policy_identity="ca-policy-v1",
            blocking_events=(),
            complete=True,
        )
        for key in keys
    }


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
        lookback_trading_sessions=1,
        estimated_rows_per_trading_day=1,
        estimated_bytes_per_row=80,
        cost_model_identity="documented-current-terms-explicit-scenario",
    )
    without_t = _midnight_frame({sep7: Decimal("100")})
    with_t = _midnight_frame({sep7: Decimal("100"), sep8: Decimal("999999")})

    # Monday first-eligible with a 1-session lookback reaches Friday, never Sunday.
    assert plan.candidates[0].start == date(2026, 9, 4)
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
        lookback_trading_sessions=1,
        estimated_rows_per_trading_day=1,
        estimated_bytes_per_row=80,
        cost_model_identity="documented-current-terms-explicit-scenario",
    )
    # Lookback reaches the sourced Thursday session; the Monday reference price
    # below proves the weekend is skipped for prior-session resolution.
    assert plan.candidates[0].start == date(2026, 9, 3)
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
        lookback_trading_sessions=1,
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
        lookback_trading_sessions=1,
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
        lookback_trading_sessions=1,
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
        lookback_trading_sessions=1,
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

    timeline = build_stage_a_prefilter_timeline(
        **base_kwargs,  # type: ignore[arg-type]
        selection_cutoffs=(d2, d4),
        corporate_action_claims_by_cutoff={
            d2: _ca_claims((KEY_A, KEY_B), as_of=d2, coverage_start=d1, coverage_end=d2),
            d4: _ca_claims((KEY_A, KEY_B), as_of=d4, coverage_start=d1, coverage_end=d4),
        },
    )
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
        prefilter=timeline[1],
        interval_minutes=5,
        expected_rows_per_trading_day=75,
        estimated_bytes_per_row=80,
        rate_limit=_rate_limit(),
        window_cutoffs=(d4,),
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
        lookback_trading_sessions=1,
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
        lookback_trading_sessions=1,
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


def _stage_a_plan_for(
    tmp_path: Path,
    dates: tuple[date, ...],
    rows_by_date: dict[date, list[dict[str, object]]],
    boundary: tuple[date, date] | None = None,
):  # type: ignore[no-untyped-def]
    manifest = _write_manifest(tmp_path, dates, rows_by_date)
    return build_stage_a_plan(
        universe_manifest_path=manifest,
        boundary=PITResearchBoundary(
            start=boundary[0] if boundary else dates[0], end=boundary[1] if boundary else dates[-1]
        ),
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


def _single_kwargs(plan, frames, keys):  # type: ignore[no-untyped-def]
    return {
        "stage_a_plan": plan,
        "daily_frames": frames,
        "daily_dataset_fingerprints": dict.fromkeys(keys, "d" * 64),
        "tick_policies": {
            key: FixedTickSizePolicy(tick_size_rupees=Decimal("0.05"), source="s") for key in keys
        },
        "corporate_actions": {
            key: CorporateActionAssessment(complete=True, blocking_events=()) for key in keys
        },
        "minimum_tradable_quantities": dict.fromkeys(keys, 1),
        "cost_provider": CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
    }


def test_lookback_requires_full_sourced_sessions_fail_closed(tmp_path: Path) -> None:
    sep7, sep8 = date(2026, 9, 7), date(2026, 9, 8)
    manifest = _write_manifest(
        tmp_path,
        (sep7, sep8),
        {sep7: [_row(sep7, KEY_A, "OPEN")], sep8: [_row(sep8, KEY_A, "OPEN")]},
        prepend_history=False,
    )
    with pytest.raises(PITAcquisitionError, match="lookback"):
        build_stage_a_plan(
            universe_manifest_path=manifest,
            boundary=PITResearchBoundary(start=sep7, end=sep8),
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


def test_eligibility_gap_raw_bars_removed_by_mandatory_mask(tmp_path: Path) -> None:
    mon, tue, wed, thu = (date(2026, 9, 7), date(2026, 9, 8), date(2026, 9, 9), date(2026, 9, 10))
    plan = _stage_a_plan_for(
        tmp_path,
        (mon, tue, wed, thu),
        {
            mon: [_row(mon, KEY_A, "OPEN")],
            tue: [_row(tue, KEY_A, "OPEN", eligible=False)],
            wed: [_row(wed, KEY_A, "OPEN")],
            thu: [_row(thu, KEY_A, "OPEN")],
        },
    )
    frames = {
        KEY_A: _midnight_frame(
            {mon: Decimal("100"), tue: Decimal("101"), wed: Decimal("102"), thu: Decimal("103")}
        )
    }
    prefilter = build_stage_a_prefilter(
        **_single_kwargs(plan, frames, (KEY_A,)),
        selection_cutoff=thu,  # type: ignore[arg-type]
        corporate_action_claims=_ca_claims(
            (KEY_A,), as_of=thu, coverage_start=mon, coverage_end=thu
        ),
    )
    assert prefilter.decisions[0].eligible is True
    stage_b = build_stage_b_plan(
        stage_a_plan=plan,
        prefilter=prefilter,
        interval_minutes=5,
        expected_rows_per_trading_day=75,
        estimated_bytes_per_row=80,
        rate_limit=_rate_limit(),
        window_cutoffs=(thu,),
    )
    detail = stage_b.details[0]
    assert detail.data_class == RAW_ACQUISITION_ONLY
    assert detail.raw_start == mon and detail.raw_end == thu
    assert detail.eligible_dates == (mon, wed, thu)
    assert detail.eligible_intervals == ((mon, mon), (wed, thu))
    # The raw range physically contains Tuesday; the mask must remove it.
    raw = _midnight_frame(
        {
            mon: Decimal("100"),
            tue: Decimal("999999"),
            wed: Decimal("102"),
            thu: Decimal("103"),
        }
    )
    research, verified = consume_research_bars(
        raw, detail, plan=stage_b, stage_a_plan=plan, trusted_prefilters=(prefilter,)
    )
    assert tuple(research.index.date) == (mon, wed, thu)
    assert (research["close"] == 999999).sum() == 0
    assert verified.data_class == "RESEARCH_READY"
    assert verified.eligible_dates == (mon, wed, thu)


def test_consumer_without_mask_fails_closed(tmp_path: Path) -> None:
    frame = _midnight_frame({date(2026, 9, 7): Decimal("100")})
    with pytest.raises(TypeError):
        consume_research_bars(frame)  # type: ignore[call-arg]
    mon, tue = date(2026, 9, 7), date(2026, 9, 8)
    plan = _stage_a_plan_for(
        tmp_path, (mon, tue), {mon: [_row(mon, KEY_A, "OPEN")], tue: [_row(tue, KEY_A, "OPEN")]}
    )
    frames = {KEY_A: _midnight_frame({mon: Decimal("100"), tue: Decimal("101")})}
    prefilter = build_stage_a_prefilter(
        **_single_kwargs(plan, frames, (KEY_A,)),
        selection_cutoff=tue,  # type: ignore[arg-type]
        corporate_action_claims=_ca_claims(
            (KEY_A,), as_of=tue, coverage_start=mon, coverage_end=tue
        ),
    )
    stage_b = build_stage_b_plan(
        stage_a_plan=plan,
        prefilter=prefilter,
        interval_minutes=5,
        expected_rows_per_trading_day=75,
        estimated_bytes_per_row=80,
        rate_limit=_rate_limit(),
        window_cutoffs=(tue,),
    )
    thin = _midnight_frame({mon: Decimal("100")})
    with pytest.raises(PITAcquisitionError, match="missing eligible observations"):
        consume_research_bars(
            thin,
            stage_b.details[0],
            plan=stage_b,
            stage_a_plan=plan,
            trusted_prefilters=(prefilter,),
        )


def test_multi_window_cannot_use_single_final_cutoff(tmp_path: Path) -> None:
    d1, d2 = date(2026, 9, 7), date(2026, 9, 8)
    plan = _stage_a_plan_for(
        tmp_path, (d1, d2), {d1: [_row(d1, KEY_A, "OPEN")], d2: [_row(d2, KEY_A, "OPEN")]}
    )
    frames = {KEY_A: _midnight_frame({d1: Decimal("100"), d2: Decimal("101")})}
    prefilter = build_stage_a_prefilter(
        **_single_kwargs(plan, frames, (KEY_A,)),
        selection_cutoff=d2,  # type: ignore[arg-type]
    )
    with pytest.raises(IncompletePrefilterError, match="timeline"):
        build_stage_b_plan(
            stage_a_plan=plan,
            prefilter=prefilter,
            interval_minutes=5,
            expected_rows_per_trading_day=75,
            estimated_bytes_per_row=80,
            rate_limit=_rate_limit(),
            window_cutoffs=(d1, d2),
        )


def test_future_ca_evidence_rejected_for_earlier_cutoff(tmp_path: Path) -> None:
    d1, d2 = date(2026, 9, 7), date(2026, 9, 8)
    plan = _stage_a_plan_for(
        tmp_path, (d1, d2), {d1: [_row(d1, KEY_A, "OPEN")], d2: [_row(d2, KEY_A, "OPEN")]}
    )
    frames = {KEY_A: _midnight_frame({d1: Decimal("100"), d2: Decimal("101")})}
    future_claim = CorporateActionEvidenceClaim(
        instrument_key=KEY_A,
        assessment_as_of=date(2026, 9, 10),
        coverage_start=d1,
        coverage_end=d2,
        source_fingerprint="e" * 64,
        policy_identity="ca-policy-v1",
        blocking_events=(),
        complete=True,
    )
    result = build_stage_a_prefilter(
        **_single_kwargs(plan, frames, (KEY_A,)),  # type: ignore[arg-type]
        selection_cutoff=d2,
        corporate_action_claims={KEY_A: future_claim},
    )
    assert result.decisions[0].eligible is False
    assert any("after decision cutoff" in failure for failure in result.failures)
    assert result.decisions[0].ca_source_fingerprint == "e" * 64


def test_delist_relist_union_and_intervals(tmp_path: Path) -> None:
    d1, d2, d3, d4, d5 = (
        date(2026, 9, 7),
        date(2026, 9, 8),
        date(2026, 9, 9),
        date(2026, 9, 10),
        date(2026, 9, 11),
    )
    assert canonical_eligible_intervals((d1, d2, d5)) == ((d1, d2), (d5, d5))
    plan = _stage_a_plan_for(
        tmp_path,
        (d1, d2, d3, d4, d5),
        {
            d1: [_row(d1, KEY_A, "OPEN"), _row(d1, KEY_B, "STEADY")],
            d2: [_row(d2, KEY_A, "OPEN", eligible=False), _row(d2, KEY_B, "STEADY")],
            d3: [_row(d3, KEY_A, "OPEN", eligible=False), _row(d3, KEY_B, "STEADY")],
            d4: [_row(d4, KEY_A, "OPEN", eligible=False), _row(d4, KEY_B, "STEADY")],
            d5: [_row(d5, KEY_A, "OPEN"), _row(d5, KEY_B, "STEADY")],
        },
    )
    frames = {
        KEY_A: _midnight_frame(
            {
                d1: Decimal("100"),
                d2: Decimal("101"),
                d3: Decimal("102"),
                d4: Decimal("103"),
                d5: Decimal("104"),
            }
        ),
        KEY_B: _midnight_frame(
            {
                d1: Decimal("100"),
                d2: Decimal("101"),
                d3: Decimal("102"),
                d4: Decimal("103"),
                d5: Decimal("104"),
            }
        ),
    }
    kwargs = _single_kwargs(plan, frames, (KEY_A, KEY_B))
    early = build_stage_a_prefilter(**kwargs, selection_cutoff=d2)  # type: ignore[arg-type]
    assert {d.instrument_key: d.eligible for d in early.decisions} == {
        KEY_A: True,
        KEY_B: True,
    }
    timeline = build_stage_a_prefilter_timeline(
        **kwargs,  # type: ignore[arg-type]
        selection_cutoffs=(d2, d5),
        corporate_action_claims_by_cutoff={
            d2: _ca_claims((KEY_A, KEY_B), as_of=d2, coverage_start=d1, coverage_end=d2),
            d5: _ca_claims((KEY_A, KEY_B), as_of=d5, coverage_start=d1, coverage_end=d5),
        },
    )
    union = build_stage_b_plan_from_timeline(
        stage_a_plan=plan,
        prefilters=timeline,
        interval_minutes=5,
        expected_rows_per_trading_day=75,
        estimated_bytes_per_row=80,
        rate_limit=_rate_limit(),
    )
    by_detail = {item.instrument_key: item for item in union.details}
    # Relisted names stay acquired for the early window that authorized them.
    assert by_detail[KEY_A].eligible_dates == (d1,)
    assert by_detail[KEY_A].eligible_intervals == ((d1, d1),)
    assert by_detail[KEY_A].authorizing_cutoffs == (d2,)
    assert by_detail[KEY_B].eligible_dates == (d1, d2, d3, d4, d5)


def test_symbol_lineage_preserved_in_stage_b(tmp_path: Path) -> None:
    d1, d2 = date(2026, 9, 7), date(2026, 9, 8)
    plan = _stage_a_plan_for(
        tmp_path,
        (d1, d2),
        {d1: [_row(d1, KEY_A, "OLDNAME")], d2: [_row(d2, KEY_A, "NEWNAME")]},
    )
    frames = {KEY_A: _midnight_frame({d1: Decimal("100"), d2: Decimal("101")})}
    kwargs = _single_kwargs(plan, frames, (KEY_A,))
    claims = {d2: _ca_claims((KEY_A,), as_of=d2, coverage_start=d1, coverage_end=d2)}
    timeline = build_stage_a_prefilter_timeline(  # type: ignore[arg-type]
        **kwargs, selection_cutoffs=(d2,), corporate_action_claims_by_cutoff=claims
    )
    union = build_stage_b_plan_from_timeline(
        stage_a_plan=plan,
        prefilters=timeline,
        interval_minutes=5,
        expected_rows_per_trading_day=75,
        estimated_bytes_per_row=80,
        rate_limit=_rate_limit(),
    )
    detail = union.details[0]
    assert detail.symbol_lineage == ((d1, "OLDNAME"), (d2, "NEWNAME"))
    assert detail.symbol_ranges == ((d1, d1, "OLDNAME"), (d2, d2, "NEWNAME"))
    assert detail.download_symbol == "NEWNAME"
    assert symbol_ranges_for(((d1, "X"), (d2, "X"))) == ((d1, d2, "X"),)


def test_stage_b_details_deterministic_fingerprint(tmp_path: Path) -> None:
    d1, d2 = date(2026, 9, 7), date(2026, 9, 8)
    plan = _stage_a_plan_for(
        tmp_path, (d1, d2), {d1: [_row(d1, KEY_A, "OPEN")], d2: [_row(d2, KEY_A, "OPEN")]}
    )
    frames = {KEY_A: _midnight_frame({d1: Decimal("100"), d2: Decimal("101")})}
    kwargs = _single_kwargs(plan, frames, (KEY_A,))
    claims = {d2: _ca_claims((KEY_A,), as_of=d2, coverage_start=d1, coverage_end=d2)}
    first_timeline = build_stage_a_prefilter_timeline(  # type: ignore[arg-type]
        **kwargs, selection_cutoffs=(d2,), corporate_action_claims_by_cutoff=claims
    )
    second_timeline = build_stage_a_prefilter_timeline(  # type: ignore[arg-type]
        **kwargs, selection_cutoffs=(d2,), corporate_action_claims_by_cutoff=claims
    )
    first = build_stage_b_plan_from_timeline(
        stage_a_plan=plan,
        prefilters=first_timeline,
        interval_minutes=5,
        expected_rows_per_trading_day=75,
        estimated_bytes_per_row=80,
        rate_limit=_rate_limit(),
    )
    second = build_stage_b_plan_from_timeline(
        stage_a_plan=plan,
        prefilters=second_timeline,
        interval_minutes=5,
        expected_rows_per_trading_day=75,
        estimated_bytes_per_row=80,
        rate_limit=_rate_limit(),
    )
    assert first.fingerprint == second.fingerprint
    assert first.details[0].data_class == RAW_ACQUISITION_ONLY
    assert first.details[0].membership_fingerprint == plan.source.manifest_sha256
    assert first.details[0].authorizing_prefilter_fingerprints == (first_timeline[0].fingerprint,)
    # Canonical gates fail closed at construction: a stale download symbol cannot
    # even be built, while a non-derived plan field still changes identity.
    with pytest.raises(PITAcquisitionError, match="canonical date-scoped lineage"):
        replace(first.details[0], download_symbol="TAMPERED")
    assert replace(first, expected_rows_per_trading_day=76).fingerprint != first.fingerprint


def _exploit_fixture(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Genuine trusted timeline with KEY_A eligible exactly on D1, D2, D5."""
    d1, d2, d3, d4, d5, d6 = (
        date(2026, 9, 7),
        date(2026, 9, 8),
        date(2026, 9, 9),
        date(2026, 9, 10),
        date(2026, 9, 11),
        date(2026, 9, 14),
    )
    dates = (d1, d2, d3, d4, d5, d6)
    plan = _stage_a_plan_for(
        tmp_path,
        dates,
        {
            d1: [_row(d1, KEY_A, "OPEN"), _row(d1, KEY_B, "STEADY")],
            d2: [_row(d2, KEY_A, "OPEN"), _row(d2, KEY_B, "STEADY")],
            d3: [_row(d3, KEY_A, "OPEN", eligible=False), _row(d3, KEY_B, "STEADY")],
            d4: [_row(d4, KEY_A, "OPEN", eligible=False), _row(d4, KEY_B, "STEADY")],
            d5: [_row(d5, KEY_A, "OPEN"), _row(d5, KEY_B, "STEADY")],
            d6: [_row(d6, KEY_A, "OPEN", eligible=False), _row(d6, KEY_B, "STEADY")],
        },
    )
    frames = {
        KEY_A: _midnight_frame(
            {d1: Decimal("100"), d2: Decimal("101"), d5: Decimal("102"), d6: Decimal("103")}
        ),
        KEY_B: _midnight_frame(
            {
                d1: Decimal("100"),
                d2: Decimal("101"),
                d3: Decimal("102"),
                d4: Decimal("103"),
                d5: Decimal("104"),
                d6: Decimal("105"),
            }
        ),
    }
    kwargs = _single_kwargs(plan, frames, (KEY_A, KEY_B))
    timeline = build_stage_a_prefilter_timeline(
        **kwargs,  # type: ignore[arg-type]
        selection_cutoffs=(d2, d6),
        corporate_action_claims_by_cutoff={
            d2: _ca_claims((KEY_A, KEY_B), as_of=d2, coverage_start=d1, coverage_end=d2),
            d6: _ca_claims((KEY_A, KEY_B), as_of=d6, coverage_start=d1, coverage_end=d6),
        },
    )
    union = build_stage_b_plan_from_timeline(
        stage_a_plan=plan,
        prefilters=timeline,
        interval_minutes=5,
        expected_rows_per_trading_day=75,
        estimated_bytes_per_row=80,
        rate_limit=_rate_limit(),
    )
    by_detail = {item.instrument_key: item for item in union.details}
    assert by_detail[KEY_A].eligible_dates == (d1, d2, d5)
    return plan, timeline, union, by_detail, (d1, d2, d3, d4, d5, d6)


def _forged_a_detail(genuine, plan, d3):  # type: ignore[no-untyped-def]
    """Self-consistent forgery injecting ineligible D3 with a recomputed mask."""
    forged_dates = (
        genuine.eligible_dates[0],
        genuine.eligible_dates[1],
        d3,
        genuine.eligible_dates[2],
    )
    lineage = tuple(sorted(genuine.symbol_lineage + ((d3, "OPEN"),)))
    return replace(
        genuine,
        eligible_dates=forged_dates,
        eligible_intervals=canonical_eligible_intervals(forged_dates),
        eligibility_mask_fingerprint=stage_b_eligibility_mask_fingerprint(
            genuine.instrument_key, forged_dates, genuine.membership_fingerprint
        ),
        symbol_lineage=lineage,
        symbol_ranges=symbol_ranges_for(lineage),
    )


def test_forged_self_consistent_detail_rejected_by_trusted_revalidation(
    tmp_path: Path,
) -> None:
    plan, timeline, union, by_detail, dates = _exploit_fixture(tmp_path)
    d1, d2, d3, _d4, _d5, _d6 = dates
    forged_a = _forged_a_detail(by_detail[KEY_A], plan, d3)
    # The forgery is internally self-consistent, so construction alone passes.
    assert forged_a.eligible_dates == (d1, d2, d3, dates[4])
    forged_plan = replace(
        union,
        details=(forged_a, by_detail[KEY_B]),
    )
    with pytest.raises(PITAcquisitionError, match="revalidation mismatch"):
        forged_plan.validate_against_prefilter_timeline(
            stage_a_plan=plan, trusted_prefilters=timeline
        )
    # And it can never enter a verified research result.
    raw = _midnight_frame(
        {d1: Decimal("100"), d2: Decimal("101"), d3: Decimal("999999"), dates[4]: Decimal("102")}
    )
    with pytest.raises(PITAcquisitionError, match="revalidation"):
        consume_research_bars(
            raw, forged_a, plan=forged_plan, stage_a_plan=plan, trusted_prefilters=timeline
        )


def test_verified_consumption_excludes_injected_gap_bar(tmp_path: Path) -> None:
    plan, timeline, union, by_detail, dates = _exploit_fixture(tmp_path)
    d1, d2, d3, _d4, d5, _d6 = dates
    raw = _midnight_frame(
        {d1: Decimal("100"), d2: Decimal("101"), d3: Decimal("999999"), d5: Decimal("102")}
    )
    research, verified = consume_research_bars(
        raw,
        by_detail[KEY_A],
        plan=union,
        stage_a_plan=plan,
        trusted_prefilters=timeline,
    )
    assert tuple(research.index.date) == (d1, d2, d5)
    assert (research["close"] == 999999).sum() == 0
    assert verified.data_class == "RESEARCH_READY"
    assert verified.research_bars_fingerprint == _filtered_bars_fingerprint(research)
    assert verified.research_bars_fingerprint != _filtered_bars_fingerprint(raw)


def test_fake_mask_hash_rejected_at_construction(tmp_path: Path) -> None:
    _plan_obj, _timeline, union, by_detail, _dates = _exploit_fixture(tmp_path)
    with pytest.raises(PITAcquisitionError, match="mask fingerprint mismatch"):
        replace(by_detail[KEY_A], eligibility_mask_fingerprint="f" * 64)


def test_changed_cutoff_rejected_by_revalidation(tmp_path: Path) -> None:
    plan, timeline, union, _by_detail, _dates = _exploit_fixture(tmp_path)
    with pytest.raises(PITAcquisitionError, match="cutoff mismatch"):
        union.validate_against_prefilter_timeline(
            stage_a_plan=plan, trusted_prefilters=(timeline[0],)
        )


def test_changed_prefilter_fingerprint_rejected(tmp_path: Path) -> None:
    plan, timeline, union, _by_detail, _dates = _exploit_fixture(tmp_path)
    tampered = replace(timeline[1], stage_a_plan_fingerprint="0" * 64)
    with pytest.raises(PITAcquisitionError, match="not bound"):
        union.validate_against_prefilter_timeline(
            stage_a_plan=plan, trusted_prefilters=(timeline[0], tampered)
        )
    # A differently-valued but well-bound prefilter still fails fingerprint agreement.
    d1, d2, d3, d4, d5, d6 = _dates
    alt_frames = {
        KEY_A: _midnight_frame(
            {d1: Decimal("100"), d2: Decimal("101"), d5: Decimal("103"), d6: Decimal("104")}
        ),
        KEY_B: _midnight_frame(
            {
                d1: Decimal("100"),
                d2: Decimal("101"),
                d3: Decimal("102"),
                d4: Decimal("103"),
                d5: Decimal("104"),
                d6: Decimal("105"),
            }
        ),
    }
    alt_d6 = build_stage_a_prefilter(
        **_single_kwargs(plan, alt_frames, (KEY_A, KEY_B)),  # type: ignore[arg-type]
        selection_cutoff=d6,
        corporate_action_claims=_ca_claims(
            (KEY_A, KEY_B), as_of=d6, coverage_start=d1, coverage_end=d6
        ),
    )
    assert alt_d6.fingerprint != timeline[1].fingerprint
    with pytest.raises(PITAcquisitionError, match="fingerprint mismatch"):
        union.validate_against_prefilter_timeline(
            stage_a_plan=plan, trusted_prefilters=(timeline[0], alt_d6)
        )


def test_changed_membership_fingerprint_rejected(tmp_path: Path) -> None:
    plan, _timeline, union, by_detail, _dates = _exploit_fixture(tmp_path)
    with pytest.raises(ValueError, match="membership fingerprint mismatch"):
        replace(union, source_manifest_sha256="a" * 64)
    assert by_detail[KEY_A].membership_fingerprint == plan.source.manifest_sha256


def test_single_cutoff_rename_uses_lineage_symbol(tmp_path: Path) -> None:
    d1, d2 = date(2026, 9, 7), date(2026, 9, 8)
    plan = _stage_a_plan_for(
        tmp_path,
        (d1, d2),
        {d1: [_row(d1, KEY_A, "OLDNAME")], d2: [_row(d2, KEY_A, "NEWNAME")]},
    )
    frames = {KEY_A: _midnight_frame({d1: Decimal("100"), d2: Decimal("101")})}
    prefilter = build_stage_a_prefilter(
        **_single_kwargs(plan, frames, (KEY_A,)),  # type: ignore[arg-type]
        selection_cutoff=d2,
        corporate_action_claims=_ca_claims((KEY_A,), as_of=d2, coverage_start=d1, coverage_end=d2),
    )
    # The decision itself still carries the prior-session symbol...
    assert prefilter.decisions[0].symbol == "OLDNAME"
    stage_b = build_stage_b_plan(
        stage_a_plan=plan,
        prefilter=prefilter,
        interval_minutes=5,
        expected_rows_per_trading_day=75,
        estimated_bytes_per_row=80,
        rate_limit=_rate_limit(),
        window_cutoffs=(d2,),
    )
    # ...but the download label must follow the canonical lineage, not the stale one.
    assert stage_b.candidates[0].symbol == "NEWNAME"
    assert stage_b.details[0].download_symbol == "NEWNAME"
    assert stage_b.details[0].symbol_lineage == ((d1, "OLDNAME"), (d2, "NEWNAME"))


def test_raw_frame_never_research_ready_without_trusted_detail(tmp_path: Path) -> None:
    plan, timeline, union, by_detail, dates = _exploit_fixture(tmp_path)
    raw = _midnight_frame({day: Decimal("100") for day in dates})
    with pytest.raises(TypeError):
        consume_research_bars(raw)  # type: ignore[call-arg]
    with pytest.raises(PITAcquisitionError, match="requires prefilters"):
        consume_research_bars(
            raw, by_detail[KEY_A], plan=union, stage_a_plan=plan, trusted_prefilters=()
        )


def test_no_live_or_order_capability(tmp_path: Path) -> None:
    plan, timeline, union, _by_detail, _dates = _exploit_fixture(tmp_path)
    assert union.as_dict()["summary"]["live_orders_called"] is False
    assert plan.as_dict()["live_orders_called"] is False
    assert timeline[0].as_dict()["live_orders_called"] is False
    _research, verified = consume_research_bars(
        _midnight_frame({day: Decimal("100") for day in _dates}),
        union.details[0],
        plan=union,
        stage_a_plan=plan,
        trusted_prefilters=timeline,
    )
    assert "live_orders_called" not in verified.as_dict()
    assert "order" not in json.dumps(verified.as_dict()).lower()
