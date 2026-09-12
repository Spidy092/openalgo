"""Build the deterministic, no-candle-download historical pilot evidence pack.

This builder consumes captured non-candle source evidence and delegates all domain
validation to the canonical calendar, PIT, session, and acquisition-plan types.
It never reads credentials, calls Upstox historical candles, or reaches an order API.
"""

from __future__ import annotations

import json
from datetime import date, time
from pathlib import Path
from typing import Any

from equity_engine.historical_acquisition_plan import (
    CorporateActionEvidence,
    EvidenceSourceIdentity,
    FormationBoundaryAdapter,
    SessionEvidence,
    aggregate_pit_evidence_fingerprint,
    build_historical_acquisition_plan,
)
from equity_engine.historical_validation import nse_session_rules_for_calendar
from equity_engine.market_sessions import (
    NSE_CAS_CONTINUOUS_END,
    NSE_CAS_EFFECTIVE_DATE,
    NSE_CAS_SOURCE,
    NSE_NORMAL_CONTINUOUS_END,
    NSE_NORMAL_CONTINUOUS_START,
)
from equity_engine.nse_calendar import nse_cm_normal_session_calendar
from equity_engine.provenance import bytes_sha256, canonical_sha256
from equity_engine.research_window_compiler import (
    FrozenTrainUniverse,
    PITMembershipSegment,
    derive_population_fingerprint,
    resolve_pit_segment,
)
from equity_engine.upstox_batch_history import HistoricalAcquisitionEvidence

PACK_ROOT = (
    Path(__file__).resolve().parents[1] / "pilot_evidence" / "historical_pilot_evidence_pack_v1"
)
RESEARCH_START = date(2026, 7, 30)
RESEARCH_END = date(2026, 8, 3)
PLANNER_SPECIAL_DATE = date(2026, 11, 8)
INSTRUMENT_KEY = "NSE_EQ|INE745G01043"
SYMBOL = "MCX"
TIMEZONE = "Asia/Kolkata"
INTERVAL_MINUTES = 5
ESTIMATED_BYTES_PER_ROW = 160

CALENDAR_SOURCE_ID = "nse-cm-calendar:2026"
SESSION_SOURCE_ID = "nse-cm-session-policy:cas-effective-2026-08-03"
BOD_SOURCE_ID = "upstox-bod:nse-eq:cas-eligibility:2026-09-12"
CA_SOURCE_ID = "nse-corporate-actions:equities:2026-07-30:2026-08-03"

MII_SOURCE = {
    "2026-07-30": {
        "source_id": "nse-mii-pit-snapshot:2026-07-30",
        "url": "https://nsearchives.nseindia.com/content/cm/NSE_CM_security_30072026.csv.gz",
        "payload_sha256": "e991c7bec6bbd1d7a51aabaeeb0359632eec5de00f3bf6f4e0aee8de60590001",
        "payload_bytes": 1209661,
        "source_row_number": 24116,
        "raw_fields": {
            "FinInstrmId": "31181",
            "TckrSymb": "MCX",
            "SctySrs": "EQ",
            "ISIN": "INE745G01043",
            "PrtdToTrad": "1",
            "SctyStsNrmlMkt": "6",
            "ElgbltyNrmlMkt": "1",
            "CallAuctnInd": "1",
            "ElgbltyClsgAuctnSsn": None,
            "NewBrdLotQty": "1",
            "BidIntrvl": "10",
            "SctyTpFlg": "0",
            "DelFlg": "N",
        },
    },
    "2026-07-31": {
        "source_id": "nse-mii-pit-snapshot:2026-07-31",
        "url": "https://nsearchives.nseindia.com/content/cm/NSE_CM_security_31072026.csv.gz",
        "payload_sha256": "5c3b3113e7c147a5be79a725f025b53b5d16f8f0826fc9ba5d8ed0198a6ef8d7",
        "payload_bytes": 1210241,
        "source_row_number": 24120,
        "raw_fields": {
            "FinInstrmId": "31181",
            "TckrSymb": "MCX",
            "SctySrs": "EQ",
            "ISIN": "INE745G01043",
            "PrtdToTrad": "1",
            "SctyStsNrmlMkt": "6",
            "ElgbltyNrmlMkt": "1",
            "CallAuctnInd": "1",
            "ElgbltyClsgAuctnSsn": "1",
            "NewBrdLotQty": "1",
            "BidIntrvl": "10",
            "SctyTpFlg": "0",
            "DelFlg": "N",
        },
    },
    "2026-08-03": {
        "source_id": "nse-mii-pit-snapshot:2026-08-03",
        "url": "https://nsearchives.nseindia.com/content/cm/NSE_CM_security_03082026.csv.gz",
        "payload_sha256": "c61c505973befcf041acd8900339b6c3b595aae18283726b4725661c29be8359",
        "payload_bytes": 1209602,
        "source_row_number": 24122,
        "raw_fields": {
            "FinInstrmId": "31181",
            "TckrSymb": "MCX",
            "SctySrs": "EQ",
            "ISIN": "INE745G01043",
            "PrtdToTrad": "1",
            "SctyStsNrmlMkt": "6",
            "ElgbltyNrmlMkt": "1",
            "CallAuctnInd": "1",
            "ElgbltyClsgAuctnSsn": "1",
            "NewBrdLotQty": "1",
            "BidIntrvl": "10",
            "SctyTpFlg": "0",
            "DelFlg": "N",
        },
    },
}

BOD_SOURCE = {
    "source_id": BOD_SOURCE_ID,
    "url": "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
    "payload_sha256": "598edaa5403775660a28ea1933218e043c5d283255a483f8105e6348bb67b2f1",
    "payload_bytes": 1940184,
    "instrument_record": {
        "segment": "NSE_EQ",
        "instrument_type": "EQ",
        "instrument_key": INSTRUMENT_KEY,
        "isin": "INE745G01043",
        "trading_symbol": SYMBOL,
        "name": "MULTI COMMODITY EXCHANGE",
        "security_type": "NORMAL",
        "cas_eligible": True,
    },
    "use_boundary": (
        "CAS eligibility only from the current BOD record; never used to back-project dated "
        "listing, MIS, BOD, suspension, or normal-market membership"
    ),
}

CA_SOURCE = {
    "source_id": CA_SOURCE_ID,
    "url": (
        "https://www.nseindia.com/api/corporates-corporateActions?index=equities&"
        "from_date=30-07-2026&to_date=03-08-2026"
    ),
    "http_status": 200,
    "response_sha256": "25414df6bdcd16634a6fb1ff7f1fc02750443762341882beefaddc6735468342",
    "response_bytes": 25440,
    "response_rows": 85,
    "coverage_start": RESEARCH_START.isoformat(),
    "coverage_end": RESEARCH_END.isoformat(),
    "exact_instrument_events": [],
    "same_symbol_events": [],
    "unknown": False,
    "complete": True,
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return bytes_sha256(path.read_bytes())


def _build_source_fingerprints(
    calendar: Any, session_policy_payload: dict[str, Any]
) -> tuple[str, str]:
    calendar_payload = {
        "trading_dates": [item.isoformat() for item in calendar.trading_dates],
        "holiday_dates": [item.isoformat() for item in calendar.holiday_dates],
        "excluded_special_session_dates": [
            item.isoformat() for item in calendar.excluded_special_session_dates
        ],
        "source_urls": list(calendar.source_urls),
    }
    return canonical_sha256(calendar_payload), canonical_sha256(session_policy_payload)


def _corroborative_current_metadata(
    current_bod: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "classification": "CORROBORATIVE_CURRENT_METADATA",
        "present": current_bod is not None,
        "blocking": False,
        "used_for_historical_selection": False,
        "used_for_pit_eligibility": False,
        "used_for_historical_cas_eligibility": False,
        "used_for_historical_plan_identity": False,
        "may_change_without_historical_plan_change": True,
        "source": current_bod,
    }


def _dated_nse_candidate_eligible(
    trade_date: date,
    source: dict[str, Any],
) -> bool:
    fields = source["raw_fields"]
    common_flags = (
        fields.get("ISIN") == "INE745G01043",
        fields.get("TckrSymb") == SYMBOL,
        fields.get("SctySrs") == "EQ",
        fields.get("PrtdToTrad") == "1",
        fields.get("ElgbltyNrmlMkt") == "1",
        fields.get("SctyStsNrmlMkt") == "6",
        fields.get("DelFlg") == "N",
    )
    if not all(common_flags):
        return False
    if trade_date == NSE_CAS_EFFECTIVE_DATE:
        return fields.get("CallAuctnInd") == "1" and fields.get("ElgbltyClsgAuctnSsn") == "1"
    return True


def _build_plan(
    *,
    mii_source: dict[str, dict[str, Any]] | None = None,
    current_bod: dict[str, Any] | None = BOD_SOURCE,
) -> tuple[Any, HistoricalAcquisitionEvidence, dict[str, Any]]:
    mii_source = MII_SOURCE if mii_source is None else mii_source
    current_metadata = _corroborative_current_metadata(current_bod)
    calendar = nse_cm_normal_session_calendar(start=RESEARCH_START, end=RESEARCH_END)
    if calendar.trading_dates != (RESEARCH_START, date(2026, 7, 31), RESEARCH_END):
        raise ValueError(f"unexpected canonical pilot calendar: {calendar.trading_dates}")
    if calendar.excluded_special_session_dates:
        raise ValueError("pilot acquisition window unexpectedly contains a special session")

    special_calendar = nse_cm_normal_session_calendar(
        start=PLANNER_SPECIAL_DATE,
        end=PLANNER_SPECIAL_DATE,
    )
    if special_calendar.excluded_special_session_dates != (PLANNER_SPECIAL_DATE,):
        raise ValueError("canonical special-session evidence did not classify 2026-11-08")

    session_policy_payload = {
        "policy_id": SESSION_SOURCE_ID,
        "policy": "NSEEquitySessionPolicy",
        "timezone": TIMEZONE,
        "continuous_start": NSE_NORMAL_CONTINUOUS_START.isoformat(),
        "normal_continuous_end": NSE_NORMAL_CONTINUOUS_END.isoformat(),
        "cas_effective_date": NSE_CAS_EFFECTIVE_DATE.isoformat(),
        "cas_continuous_end": NSE_CAS_CONTINUOUS_END.isoformat(),
        "cas_auxiliary_window": ["15:15:00", "15:35:00"],
        "cas_source": NSE_CAS_SOURCE,
        "calendar_source_urls": list(calendar.source_urls),
    }
    calendar_fingerprint, session_fingerprint = _build_source_fingerprints(
        calendar, session_policy_payload
    )

    pit_segments = tuple(
        PITMembershipSegment(
            instrument_key=INSTRUMENT_KEY,
            valid_from=date.fromisoformat(trade_date),
            valid_to=date.fromisoformat(segment_end),
            evidence_as_of=date.fromisoformat(trade_date),
            source_fingerprint=str(mii_source[trade_date]["payload_sha256"]),
            eligible=True,
        )
        for trade_date, segment_end in (
            ("2026-07-30", "2026-07-30"),
            ("2026-07-31", "2026-08-02"),
            ("2026-08-03", "2026-08-03"),
        )
    )
    for trade_date in calendar.trading_dates:
        resolved = resolve_pit_segment(pit_segments, trade_date)
        if resolved.evidence_as_of > trade_date or not resolved.eligible:
            raise ValueError(f"PIT proof failed for {trade_date}")

    ca_identity_payload = {
        "source_id": CA_SOURCE_ID,
        "response_sha256": CA_SOURCE["response_sha256"],
        "query": {
            "start": RESEARCH_START.isoformat(),
            "end": RESEARCH_END.isoformat(),
            "instrument_key": INSTRUMENT_KEY,
            "symbol": SYMBOL,
        },
        "coverage": {
            "complete": True,
            "unknown": False,
            "exact_instrument_events": [],
            "same_symbol_events": [],
        },
        "policy": {
            "name": "raw-unadjusted-block-structural-actions-v1",
            "structural_event_types": [
                "Split",
                "Bonus",
                "Rights",
                "Merger",
                "Demerger",
                "Delisting",
                "Capital Reduction",
                "Other Structural",
            ],
            "unknown_is_not_no_action": True,
        },
    }
    ca_fingerprint = canonical_sha256(ca_identity_payload)
    corporate_actions = CorporateActionEvidence(
        source_id=CA_SOURCE_ID,
        fingerprint=ca_fingerprint,
        coverage_start=RESEARCH_START,
        coverage_end=RESEARCH_END,
        covered_instruments=(INSTRUMENT_KEY,),
        complete=True,
        unknown_instruments=(),
        blocking_events=(),
        policy_identity="raw-unadjusted-block-structural-actions-v1",
        events_count=0,
    )

    session_rules = nse_session_rules_for_calendar(
        calendar,
        timezone=TIMEZONE,
        interval_minutes=INTERVAL_MINUTES,
        cas_eligible=True,
    )
    cas_rule = session_rules[RESEARCH_END]
    if (
        cas_rule.end_time != time(15, 15)
        or cas_rule.auxiliary_start_time != time(15, 15)
        or cas_rule.auxiliary_end_time != time(15, 35)
    ):
        raise ValueError("canonical CAS session rule did not round-trip to 15:15/15:35")
    for trade_date in calendar.trading_dates:
        if not _dated_nse_candidate_eligible(trade_date, mii_source[trade_date.isoformat()]):
            raise ValueError(f"dated NSE CAS/listing evidence failed for {trade_date.isoformat()}")

    sessions = tuple(
        SessionEvidence(
            trade_date=trade_date,
            session_kind="NORMAL",
            source_id=SESSION_SOURCE_ID,
            timezone=TIMEZONE,
            start_time=rule.start_time.isoformat(),
            end_time_exclusive=rule.end_time.isoformat(),
            expected_rows_by_interval=(
                (INTERVAL_MINUTES, 76 if rule.auxiliary_start_time is not None else 75),
            ),
        )
        for trade_date, rule in sorted(session_rules.items())
    )

    population_policy_id = "historical-pilot-evidence-pack-v1:dated-nse-cas-pit-v2"
    population_fingerprint = derive_population_fingerprint(
        instruments=(INSTRUMENT_KEY,),
        universe_policy_id=population_policy_id,
        pit_segments=(pit_segments[0],),
    )
    frozen_universe = FrozenTrainUniverse(
        instruments=(INSTRUMENT_KEY,),
        universe_policy_id=population_policy_id,
        population_fingerprint=population_fingerprint,
        frozen_as_of=RESEARCH_START,
    )
    formation_fingerprint = canonical_sha256(
        {
            "artifact": "historical-pilot-evidence-pack-v1",
            "formation_boundary": RESEARCH_START.isoformat(),
            "frozen_population": [INSTRUMENT_KEY],
            "purpose": "acquisition-only pilot; no WFO selection performed",
        }
    )
    formation_adapter = FormationBoundaryAdapter(
        formation_boundary=RESEARCH_START,
        frozen_train_universe=frozen_universe,
        research_window_plan_fingerprint=formation_fingerprint,
    )

    evidence_sources = [
        EvidenceSourceIdentity(
            source_id=CALENDAR_SOURCE_ID,
            evidence_type="NSE CM calendar evidence",
            fingerprint=calendar_fingerprint,
        ),
        EvidenceSourceIdentity(
            source_id=SESSION_SOURCE_ID,
            evidence_type="NSE session policy and CAS evidence",
            fingerprint=session_fingerprint,
        ),
        EvidenceSourceIdentity(
            source_id=CA_SOURCE_ID,
            evidence_type="corporate-action coverage and no-action evidence",
            fingerprint=ca_fingerprint,
        ),
    ]
    evidence_sources.extend(
        EvidenceSourceIdentity(
            source_id=str(source["source_id"]),
            evidence_type="PITMembershipSegment/NSE MII security snapshot",
            fingerprint=str(source["payload_sha256"]),
        )
        for source in (mii_source[key] for key in sorted(mii_source))
    )

    plan = build_historical_acquisition_plan(
        research_start=RESEARCH_START,
        research_end=RESEARCH_END,
        calendar=calendar,
        calendar_source_id=CALENDAR_SOURCE_ID,
        evidence_sources=tuple(evidence_sources),
        pit_segments=pit_segments,
        formation_boundary_adapter=formation_adapter,
        session_evidence=sessions,
        corporate_action_evidence=corporate_actions,
        requested_interval_minutes=(INTERVAL_MINUTES,),
        estimated_bytes_per_row=ESTIMATED_BYTES_PER_ROW,
    )
    if plan.historical_acquisition_superset != (INSTRUMENT_KEY,):
        raise ValueError("unexpected acquisition superset")
    if plan.frozen_wfo_population != (INSTRUMENT_KEY,):
        raise ValueError("unexpected frozen WFO population")
    if plan.expected_request_count != 1 or plan.estimated_rows != 226:
        raise ValueError(
            f"unexpected plan estimate: requests={plan.expected_request_count} rows={plan.estimated_rows}"
        )

    pit_evidence_fingerprint = aggregate_pit_evidence_fingerprint(pit_segments)
    acquisition_evidence = HistoricalAcquisitionEvidence(
        pit_fingerprint=pit_evidence_fingerprint,
        corporate_action_fingerprint=ca_fingerprint,
        acquisition_plan_fingerprint=plan.deterministic_fingerprint(),
        session_policy_identity=SESSION_SOURCE_ID,
        expected_trade_dates=tuple(calendar.trading_dates),
        session_rules=session_rules,
    )
    return (
        plan,
        acquisition_evidence,
        {
            "calendar": calendar,
            "special_calendar": special_calendar,
            "session_policy_payload": session_policy_payload,
            "calendar_fingerprint": calendar_fingerprint,
            "session_fingerprint": session_fingerprint,
            "ca_identity_payload": ca_identity_payload,
            "ca_fingerprint": ca_fingerprint,
            "pit_segments": pit_segments,
            "pit_evidence_fingerprint": pit_evidence_fingerprint,
            "mii_source": mii_source,
            "current_metadata": current_metadata,
            "session_rules": session_rules,
        },
    )


def _candidate_manifest(plan: Any, context: dict[str, Any]) -> dict[str, Any]:
    dates = []
    for trade_date in plan.normal_trading_dates:
        source = context["mii_source"][trade_date.isoformat()]
        fields = dict(source["raw_fields"])
        dates.append(
            {
                "trade_date": trade_date.isoformat(),
                "evidence_as_of": trade_date.isoformat(),
                "exactly_one_pit_segment": sum(
                    segment.covers(trade_date) for segment in context["pit_segments"]
                )
                == 1,
                "eligible": _dated_nse_candidate_eligible(trade_date, source),
                "listed_on_nse": True,
                "normal_equity": True,
                "tradeable_in_normal_market": True,
                "current_mis_bod_suspension_backprojection": False,
                "source_url": source["url"],
                "source_payload_sha256": source["payload_sha256"],
                "source_row_number": source["source_row_number"],
                "raw_fields": fields,
                "pit_segment": next(
                    segment.as_dict()
                    for segment in context["pit_segments"]
                    if segment.covers(trade_date)
                ),
            }
        )

    manifest = {
        "schema_version": "historical-pilot-candidate-manifest/v1",
        "pack_id": "historical-pilot-evidence-pack-v1",
        "selection_rule": (
            "From the exact pilot date intersection, retain dated NSE CM EQ rows with canonical "
            "v1.5 PrtdToTrad=1, ElgbltyNrmlMkt=1, active normal-market status, exact ISIN "
            "identity, and one PITMembershipSegment per date; on 2026-08-03 additionally "
            "require dated NSE CallAuctnInd=1 and ElgbltyClsgAuctnSsn=1 under the canonical "
            "NSE closing-auction session policy. Sort instrument_key and select the first "
            "result. No current BOD, price, performance, liquidity, MIS, or suspension "
            "backprojection is used."
        ),
        "selection_inputs": {
            "normal_dates": [item.isoformat() for item in plan.normal_trading_dates],
            "cas_effective_date": NSE_CAS_EFFECTIVE_DATE.isoformat(),
            "cas_source": NSE_CAS_SOURCE,
            "dated_cas_source_id": context["mii_source"]["2026-08-03"]["source_id"],
            "pit_evidence_fingerprint": context["pit_evidence_fingerprint"],
            "max_instruments": 3,
            "max_instrument_trading_days": 7,
            "selected_instrument_trading_days": len(dates),
        },
        "selected_candidates": [
            {
                "instrument_key": INSTRUMENT_KEY,
                "symbol": SYMBOL,
                "isin": "INE745G01043",
                "selection_rank": 1,
                "role": "CAS_CONTROL",
                "dates": dates,
                "pit_evidence_fingerprint": context["pit_evidence_fingerprint"],
                "dated_nse_cas_evidence": {
                    "trade_date": RESEARCH_END.isoformat(),
                    "source": context["mii_source"][RESEARCH_END.isoformat()],
                    "session_policy_id": SESSION_SOURCE_ID,
                    "selection_flags": {
                        "CallAuctnInd": "1",
                        "ElgbltyClsgAuctnSsn": "1",
                        "ElgbltyNrmlMkt": "1",
                        "PrtdToTrad": "1",
                    },
                },
                "corroborative_current_metadata": context["current_metadata"],
                "corporate_action_evidence": {
                    **CA_SOURCE,
                    "fingerprint": context["ca_fingerprint"],
                    "policy_identity": "raw-unadjusted-block-structural-actions-v1",
                    "status": "NO_ACTION_CONFIRMED_BY_COMPLETE_COVERAGE",
                    "structural_events": [],
                    "unknown_is_not_no_action": True,
                },
                "acquisition_interval": {
                    "start": RESEARCH_START.isoformat(),
                    "end": RESEARCH_END.isoformat(),
                    "interval_minutes": INTERVAL_MINUTES,
                    "trade_dates": [item.isoformat() for item in plan.normal_trading_dates],
                },
            }
        ],
        "population_separation": {
            "historical_acquisition_superset": list(plan.historical_acquisition_superset),
            "frozen_wfo_population": list(plan.frozen_wfo_population),
            "same_membership_is_intentional": True,
            "reason": (
                "The pilot is acquisition-only and has no dynamic stock ranking; the canonical "
                "superset and formation-boundary frozen population are separate fields even "
                "though this one-instrument pilot has the same member."
            ),
        },
        "live_orders_called": False,
    }
    identity_manifest = dict(manifest)
    identity_manifest["selected_candidates"] = [
        {key: value for key, value in candidate.items() if key != "corroborative_current_metadata"}
        for candidate in manifest["selected_candidates"]
    ]
    manifest["manifest_fingerprint"] = canonical_sha256(identity_manifest)
    return manifest


def build_pack(
    output_dir: Path = PACK_ROOT,
    *,
    mii_source: dict[str, dict[str, Any]] | None = None,
    current_bod: dict[str, Any] | None = BOD_SOURCE,
) -> dict[str, Any]:
    plan, acquisition_evidence, context = _build_plan(
        mii_source=mii_source,
        current_bod=current_bod,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    plan_path = output_dir / "historical_acquisition_plan.json"
    evidence_path = output_dir / "acquisition_evidence.json"
    candidate_manifest_path = output_dir / "selected_candidate_manifest.json"
    candidate_file_path = output_dir / "candidate_file.json"
    prefilter_path = output_dir / "prefilter_evidence.json"
    source_path = output_dir / "source_evidence.json"
    dry_run_path = output_dir / "dry_run_request_plan.json"
    special_path = output_dir / "excluded_special_session.json"
    summary_path = output_dir / "pilot_summary.md"
    fingerprints_path = output_dir / "fingerprints.json"

    _write_json(plan_path, json.loads(plan.to_json()))
    evidence_payload = acquisition_evidence.as_dict()
    evidence_payload["evidence_fingerprint"] = acquisition_evidence.fingerprint()
    _write_json(evidence_path, evidence_payload)

    manifest = _candidate_manifest(plan, context)
    _write_json(candidate_manifest_path, manifest)
    candidate_payload = [
        {
            "instrument_key": INSTRUMENT_KEY,
            "symbol": SYMBOL,
            "start": RESEARCH_START.isoformat(),
            "end": RESEARCH_END.isoformat(),
        }
    ]
    _write_json(candidate_file_path, candidate_payload)

    prefilter = {
        "schema_version": "historical-pilot-prefilter-evidence/v1",
        "candidate_manifest_fingerprint": manifest["manifest_fingerprint"],
        "pit_evidence_fingerprint": context["pit_evidence_fingerprint"],
        "selection_rule": manifest["selection_rule"],
        "candidate_file": "candidate_file.json",
        "candidate_file_sha256": canonical_sha256(candidate_payload),
        "affordability_prefilter_applied": True,
        "affordability_note": "Pilot scope is deterministic and bounded; no price/performance ranking or broad acquisition was performed.",
        "historical_acquisition_superset": list(plan.historical_acquisition_superset),
        "frozen_wfo_population": list(plan.frozen_wfo_population),
        "live_orders_called": False,
    }
    prefilter["fingerprint"] = canonical_sha256(prefilter)
    _write_json(prefilter_path, prefilter)

    source_payload = {
        "schema_version": "historical-pilot-source-evidence/v1",
        "calendar": {
            "source_id": CALENDAR_SOURCE_ID,
            "fingerprint": context["calendar_fingerprint"],
            "research_window": [RESEARCH_START.isoformat(), RESEARCH_END.isoformat()],
            "normal_trading_dates": [
                item.isoformat() for item in context["calendar"].trading_dates
            ],
            "holiday_dates": [item.isoformat() for item in context["calendar"].holiday_dates],
            "source_urls": list(context["calendar"].source_urls),
        },
        "planner_special_session": {
            "trade_date": PLANNER_SPECIAL_DATE.isoformat(),
            "calendar_source_id": CALENDAR_SOURCE_ID,
            "source_urls": list(context["special_calendar"].source_urls),
            "classification": "OUTSIDE_PILOT_WINDOW",
            "session_classification": "SPECIAL_SESSION_EXCLUDED",
            "counted_as_instrument_trading_day": False,
        },
        "nse_mii_pit_snapshots": [
            context["mii_source"][key] for key in sorted(context["mii_source"])
        ],
        "pit_evidence_fingerprint": context["pit_evidence_fingerprint"],
        "corroborative_current_metadata": context["current_metadata"],
        "corporate_action_query": {
            **CA_SOURCE,
            "fingerprint": context["ca_fingerprint"],
            "identity_payload": context["ca_identity_payload"],
        },
        "session_policy": {
            **context["session_policy_payload"],
            "fingerprint": context["session_fingerprint"],
        },
        "credentials_included": False,
        "historical_candles_downloaded": False,
        "live_orders_called": False,
    }
    _write_json(source_path, source_payload)

    dry_run_payload = {
        "schema_version": "historical-pilot-dry-run-request-plan/v1",
        "mode": "DRY_RUN",
        "candidate_file": "candidate_file.json",
        "interval_minutes": INTERVAL_MINUTES,
        "requests": [
            {
                "instrument_key": INSTRUMENT_KEY,
                "symbol": SYMBOL,
                "start": RESEARCH_START.isoformat(),
                "end": RESEARCH_END.isoformat(),
                "trade_dates": [item.isoformat() for item in plan.normal_trading_dates],
                "chunk_count": 1,
            }
        ],
        "estimated_request_count": plan.expected_request_count,
        "estimated_rows": plan.estimated_rows,
        "estimated_raw_storage_bytes": plan.estimated_storage_bytes,
        "estimated_bytes_per_row": plan.estimated_bytes_per_row,
        "pit_evidence_fingerprint": context["pit_evidence_fingerprint"],
        "cas_estimate": {
            "continuous_rows": 72,
            "auxiliary_rows_upper_bound": 4,
            "raw_rows_used_for_estimate": 76,
            "continuous_end": "15:15:00",
            "auxiliary_window": ["15:15:00", "15:35:00"],
        },
        "safe_chunk_calendar_days": plan.safe_chunk_calendar_days,
        "live_orders_called": False,
    }
    _write_json(dry_run_path, dry_run_payload)

    special_payload = {
        "schema_version": "historical-pilot-special-session-exclusion/v1",
        "trade_date": PLANNER_SPECIAL_DATE.isoformat(),
        "session_kind": "SPECIAL",
        "classification": "OUTSIDE_PILOT_WINDOW",
        "session_classification": "SPECIAL_SESSION_EXCLUDED",
        "acquisition_allowed": False,
        "counted_as_instrument_trading_day": False,
        "downloaded": False,
        "timezone": TIMEZONE,
        "source_id": CALENDAR_SOURCE_ID,
        "source_urls": list(context["special_calendar"].source_urls),
        "reason": (
            "OUTSIDE_PILOT_WINDOW audit example: canonical NSE CM calendar identifies 2026-11-08 "
            "as a separately announced Muhurat session; NOT_COUNTED in this pilot, no "
            "normal-session rule is inferred, and no candle request is authorized."
        ),
        "live_orders_called": False,
    }
    _write_json(special_path, special_payload)

    plan_fingerprint = plan.deterministic_fingerprint()
    evidence_fingerprint = acquisition_evidence.fingerprint()
    pit_evidence_fingerprint = context["pit_evidence_fingerprint"]
    fingerprints = {
        "schema_version": "historical-pilot-fingerprints/v1",
        "acquisition_plan_fingerprint": plan_fingerprint,
        "acquisition_evidence_fingerprint": evidence_fingerprint,
        "plan_id": plan.plan_id,
        "candidate_manifest_fingerprint": manifest["manifest_fingerprint"],
        "prefilter_evidence_fingerprint": prefilter["fingerprint"],
        "pit_evidence_fingerprint": context["pit_evidence_fingerprint"],
        "source_fingerprints": {
            "calendar": context["calendar_fingerprint"],
            "session_policy": context["session_fingerprint"],
            "corporate_actions": context["ca_fingerprint"],
            "nse_mii": {
                key: context["mii_source"][key]["payload_sha256"]
                for key in sorted(context["mii_source"])
            },
        },
        "corroborative_current_metadata": context["current_metadata"],
        "file_sha256": {},
        "live_orders_called": False,
    }
    _write_json(fingerprints_path, fingerprints)
    for path in (
        plan_path,
        evidence_path,
        candidate_manifest_path,
        candidate_file_path,
        prefilter_path,
        source_path,
        dry_run_path,
        special_path,
    ):
        fingerprints["file_sha256"][path.name] = _file_sha256(path)
    _write_json(fingerprints_path, fingerprints)

    summary_path.write_text(
        f"""# Historical Pilot Evidence Pack V1

- **Mode:** DRY_RUN only; no historical candles were downloaded.
- **Selection:** deterministic dated NSE/PIT intersection + canonical dated CAS eligibility, sorted by `instrument_key`, first result; current BOD is nonblocking corroborative metadata and no price/performance ranking was used.
- **Selected instrument:** `{INSTRUMENT_KEY}` (`{SYMBOL}`)
- **Pilot dates:** `2026-07-30`, `2026-07-31`, `2026-08-03` (`NSE_CAS_EFFECTIVE_DATE`; CAS-aware)
- **Special-session audit example:** `2026-11-08`, `OUTSIDE_PILOT_WINDOW` / `NOT_COUNTED` / `SPECIAL_SESSION_EXCLUDED`; not part of the canonical acquisition window and not downloaded.
- **Instrument-trading-days:** `{len(plan.normal_trading_dates)}` (limit 7)
- **Acquisition requests:** `{plan.expected_request_count}`
- **Estimated rows:** `{plan.estimated_rows}` (`75 + 75 + 76`, including the CAS auxiliary upper bound)
- **Estimated raw storage:** `{plan.estimated_storage_bytes}` bytes at `{plan.estimated_bytes_per_row}` bytes/row
- **Canonical acquisition plan:** `{plan.plan_id}` / `{plan_fingerprint}`
- **Acquisition evidence fingerprint:** `{evidence_fingerprint}`
- **Aggregate PIT evidence fingerprint:** `{pit_evidence_fingerprint}` (all three canonical PIT segments)
- **Corporate-action status:** complete query coverage; `NO_ACTION_CONFIRMED_BY_COMPLETE_COVERAGE`; UNKNOWN is not treated as no action.
- **Session:** `Asia/Kolkata`, continuous `09:15`–`15:30` on normal dates; CAS continuous end `15:15`, auxiliary `15:15`–`15:35` on `2026-08-03`; source `{NSE_CAS_SOURCE}`.
- **Safety:** `live_orders_called=false`; credentials are not present in the pack.

The special-session record is intentionally a separately sourced audit example: it is outside the pilot window, marked `OUTSIDE_PILOT_WINDOW` and `NOT_COUNTED`, and is not treated as an in-window acquisition exclusion.
""",
        encoding="utf-8",
    )

    return {
        "output_dir": str(output_dir),
        "plan_id": plan.plan_id,
        "acquisition_plan_fingerprint": plan_fingerprint,
        "acquisition_evidence_fingerprint": evidence_fingerprint,
        "estimated_request_count": plan.expected_request_count,
        "estimated_rows": plan.estimated_rows,
        "estimated_storage_bytes": plan.estimated_storage_bytes,
        "live_orders_called": False,
    }


def main() -> int:
    result = build_pack()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
