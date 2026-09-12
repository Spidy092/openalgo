from __future__ import annotations

import hashlib
import json
from datetime import date, time
from pathlib import Path

import httpx
import pandas as pd
import pytest

from equity_engine.historical_validation import IntradaySessionRule
from equity_engine.market_sessions import NSEEquitySessionPolicy
from equity_engine.provenance import canonical_sha256
from equity_engine.upstox_batch_history import (
    ArtifactCorruptionError,
    HistoricalAcquisitionEvidence,
    HistoricalBatchCandidate,
    UpstoxHistoricalBatchDownloader,
    aggregate_raw_sha,
)
from equity_engine.validated_dataset_handoff import (
    HandoffValidationError,
    RawAcquisitionArtifact,
    build_validated_dataset_handoff,
    session_policy_fingerprint,
    verify_raw_acquisition,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _cas_rule() -> IntradaySessionRule:
    return IntradaySessionRule(
        rule_id="synthetic-nse-cas-session",
        timezone="Asia/Kolkata",
        start_time=time(9, 15),
        end_time=time(15, 15),
        interval_minutes=5,
        source_reference="synthetic-nse-session-evidence",
        auxiliary_start_time=time(15, 15),
        auxiliary_end_time=time(15, 35),
        auxiliary_semantics="synthetic-CAS-auxiliary",
    )


class _KiroStyleClient:
    def get(self, url: str, **_: object) -> httpx.Response:
        trade_date = date.fromisoformat(url.rsplit("/", 1)[-1])
        index = pd.date_range(
            f"{trade_date.isoformat()} 09:15",
            periods=72,
            freq="5min",
            tz="Asia/Kolkata",
        ).append(
            pd.date_range(
                f"{trade_date.isoformat()} 15:15",
                periods=4,
                freq="5min",
                tz="Asia/Kolkata",
            )
        )
        candles = [
            [timestamp.isoformat(), 100, 101, 99, 100.5, 1000 + number, 0]
            for number, timestamp in enumerate(index)
        ]
        return httpx.Response(
            200,
            json={"status": "success", "data": {"candles": candles}},
            request=httpx.Request("GET", url),
        )


def _status_variant(
    manifest_path: Path,
    *,
    status: str,
    destination: Path,
) -> Path:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    state_path = manifest_path.with_suffix(".state.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    payload["status"] = status
    identity_fields = (
        "schema_version",
        "artifact_type",
        "request",
        "pit_evidence_fingerprint",
        "corporate_action_evidence_fingerprint",
        "acquisition_plan_fingerprint",
        "session_policy_identity",
        "evidence_fingerprint",
        "adjustment_policy",
        "universe_rule_version",
        "raw_sha256",
        "raw_artifacts_identity",
        "requested_dates",
        "status",
    )
    payload["manifest_fingerprint"] = canonical_sha256(
        {key: payload[key] for key in identity_fields}
    )
    state["status"] = status
    state["failure"] = None if status == "PARTIAL" else "synthetic failed acquisition"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    destination.with_suffix(".state.json").write_text(json.dumps(state), encoding="utf-8")
    return destination


def test_kiro_acquisition_to_research_handoff_is_single_source_and_fail_closed(
    tmp_path: Path,
) -> None:
    trade_date = date(2026, 9, 8)
    instrument_key = "NSE_EQ|INE001A01036"
    policy = NSEEquitySessionPolicy(cas_eligible=True, exit_buffer_minutes=0)
    evidence = HistoricalAcquisitionEvidence(
        pit_fingerprint=_digest("integration:pit"),
        corporate_action_fingerprint=_digest("integration:corporate-actions"),
        acquisition_plan_fingerprint=_digest("integration:acquisition-plan"),
        session_policy_identity=session_policy_fingerprint(policy),
        expected_trade_dates=(trade_date,),
        session_rules={trade_date: _cas_rule()},
    )
    output_dir = tmp_path / "kiro-acquisition"
    result = UpstoxHistoricalBatchDownloader(
        access_token="integration-secret-never-persisted",
        output_dir=output_dir,
        client=_KiroStyleClient(),
        min_request_interval_seconds=0,
        max_attempts=1,
        backoff_seconds=0,
    ).run(
        candidates=[
            HistoricalBatchCandidate(
                instrument_key=instrument_key,
                symbol="OPEN",
                start=trade_date,
                end=trade_date,
            )
        ],
        universe_rule_version="integration-rule-v1",
        adjustment_policy="raw-unadjusted",
        evidence=evidence,
    )

    assert result.passed is True
    item = result.items[0]
    manifest_path = Path(item.manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = json.loads(manifest_path.with_suffix(".state.json").read_text(encoding="utf-8"))
    raw_path = Path(payload["raw_artifacts"][0]["raw_path"])
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    assert payload["status"] == state["status"] == "COMPLETE"
    assert payload["requested_dates"] == [trade_date.isoformat()]
    assert payload["covered_dates"] == [trade_date.isoformat()]
    assert payload["raw_sha256"] == state["raw_sha256"] == aggregate_raw_sha([raw_sha])
    assert payload["raw_sha256"] != payload["parquet_sha256"]
    assert payload["live_orders_called"] is False
    assert state["live_orders_called"] is False
    assert "integration-secret-never-persisted" not in manifest_path.read_text(encoding="utf-8")

    artifact = RawAcquisitionArtifact.from_kiro_manifest(manifest_path)
    assert artifact.manifest.as_dict() == payload
    assert artifact.manifest.raw_sha256 == payload["raw_sha256"]
    verified, descriptor, research_input = build_validated_dataset_handoff(
        artifact,
        expected_instrument_key=instrument_key,
        expected_interval="5m",
        session_policy=policy,
    )

    assert verified.live_orders_called is False
    assert descriptor.live_orders_called is False
    assert research_input.live_orders_called is False
    assert descriptor.raw_row_count == 76
    assert descriptor.research_row_count == 72
    assert descriptor.excluded_auxiliary_row_count == 4
    assert len(research_input.frame) == 72
    assert all(timestamp.time() < time(15, 15) for timestamp in research_input.frame.index)

    original_raw = raw_path.read_bytes()
    raw_path.write_bytes(original_raw + b"corruption")
    with pytest.raises(HandoffValidationError, match="SHA-256"):
        verify_raw_acquisition(
            artifact,
            expected_instrument_key=instrument_key,
            expected_interval="5m",
        )
    raw_path.write_bytes(original_raw)

    research_input.frame.iloc[0, research_input.frame.columns.get_loc("close")] += 1
    with pytest.raises(HandoffValidationError, match="changed after validation"):
        research_input.frame_for_research()

    for status in ("PARTIAL", "FAILED"):
        variant_path = _status_variant(
            manifest_path,
            status=status,
            destination=tmp_path / f"{status.lower()}.manifest.json",
        )
        variant = RawAcquisitionArtifact.from_kiro_manifest(variant_path)
        with pytest.raises(HandoffValidationError, match="not research-ready"):
            build_validated_dataset_handoff(
                variant,
                expected_instrument_key=instrument_key,
                expected_interval="5m",
                session_policy=policy,
            )

    changed_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed_payload["pit_evidence_fingerprint"] = _digest("integration:changed-pit")
    changed_manifest_path = tmp_path / "changed-evidence.manifest.json"
    changed_manifest_path.write_text(json.dumps(changed_payload), encoding="utf-8")
    changed_manifest_path.with_suffix(".state.json").write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ArtifactCorruptionError, match="fingerprint"):
        RawAcquisitionArtifact.from_kiro_manifest(changed_manifest_path)

    with pytest.raises(ValueError, match="access_token"):
        UpstoxHistoricalBatchDownloader(
            access_token="",
            output_dir=tmp_path / "no-credentials",
            client=_KiroStyleClient(),
        )
