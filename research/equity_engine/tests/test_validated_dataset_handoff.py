from __future__ import annotations

import hashlib
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from equity_engine.market_sessions import NSEEquitySessionPolicy
from equity_engine.provenance import MarketDataManifest, canonical_sha256, dataframe_fingerprint
from equity_engine.upstox_batch_history import aggregate_raw_sha
from equity_engine.validated_dataset_handoff import (
    AcquisitionArtifactManifest,
    AcquisitionStatus,
    HandoffValidationError,
    RawAcquisitionArtifact,
    build_continuous_session_research_input,
    build_validated_dataset_handoff,
    session_policy_fingerprint,
    verify_raw_acquisition,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            "2026-09-07 09:15",
            "2026-09-07 15:10",
            "2026-09-07 15:15",
            "2026-09-07 15:20",
        ],
        tz="Asia/Kolkata",
    )
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [100.5, 101.5, 102.5, 103.5],
            "low": [99.5, 100.5, 101.5, 102.5],
            "close": [100.2, 101.2, 102.2, 103.2],
            "volume": [1000, 1100, 1200, 1300],
        },
        index=index,
    )


def _artifact(
    tmp_path: Path,
    *,
    frame: pd.DataFrame | None = None,
    status: AcquisitionStatus = AcquisitionStatus.COMPLETE,
    instrument_key: str = "NSE_EQ|INE002A01018",
    interval: str = "5m",
    timezone: str = "Asia/Kolkata",
    pit_fingerprint: str | None = None,
    corporate_action_fingerprint: str | None = None,
    session_policy: NSEEquitySessionPolicy | None = None,
    adjustment_policy: str = "raw",
    raw_sha256: str | None = None,
) -> RawAcquisitionArtifact:
    actual_frame = _frame() if frame is None else frame
    parquet_path = tmp_path / "raw.parquet"
    actual_frame.to_parquet(parquet_path)
    raw_path = tmp_path / "2026-09-07.raw.json"
    raw_payload = b'{"status":"success","data":{"candles":[]}}'
    raw_path.write_bytes(raw_payload)
    raw_digest = hashlib.sha256(raw_payload).hexdigest()
    raw_path.with_name(raw_path.name.removesuffix(".raw.json") + ".raw.sha256").write_text(
        f"{raw_digest}\n"
    )
    policy = session_policy or NSEEquitySessionPolicy(cas_eligible=True, exit_buffer_minutes=0)
    market_manifest = MarketDataManifest(
        provider="synthetic-local-fixture",
        exchange="NSE",
        instrument_token=instrument_key,
        symbol="RELIANCE",
        timezone="Asia/Kolkata",
        interval=interval,
        timestamp_semantics="candle_start",
        start=actual_frame.index[0].to_pydatetime(),
        end=actual_frame.index[-1].to_pydatetime(),
        retrieved_at=datetime(2026, 9, 10, tzinfo=UTC),
        adjustment_policy=adjustment_policy,
        universe_rule_version="synthetic-pit-v1",
        source_reference="synthetic:local-fixture",
    )
    trade_dates = tuple(sorted(set(actual_frame.index.date)))
    pit = pit_fingerprint or _digest("pit-v1")
    corporate_actions = corporate_action_fingerprint or _digest("ca-v1")
    plan = _digest("plan-v1")
    session_identity = session_policy_fingerprint(policy)
    request = {
        "instrument_key": instrument_key,
        "symbol": "RELIANCE",
        "start": trade_dates[0].isoformat(),
        "end": trade_dates[-1].isoformat(),
        "interval_minutes": int(interval.removesuffix("m")),
    }
    chunk = {
        "schema_version": 1,
        "artifact_type": "upstox_v3_historical_raw_response",
        "request": {
            **request,
            "chunk_start": trade_dates[0].isoformat(),
            "chunk_end": trade_dates[-1].isoformat(),
        },
        "covered_dates": [day.isoformat() for day in trade_dates],
        "raw_sha256": raw_digest,
        "rows": len(actual_frame),
        "timezone": str(actual_frame.index.tz),
        "adjustment_policy": adjustment_policy,
        "pit_evidence_fingerprint": pit,
        "corporate_action_evidence_fingerprint": corporate_actions,
        "acquisition_plan_fingerprint": plan,
        "session_policy_identity": session_identity,
        "evidence_fingerprint": _digest("evidence-v1"),
        "source_api": "upstox_v3_historical_candle",
        "request_url": "synthetic:upstox-v3",
        "universe_rule_version": "synthetic-pit-v1",
        "retrieval_timestamp": datetime(2026, 9, 10, tzinfo=UTC).isoformat(),
        "raw_path": str(raw_path),
        "raw_bytes": len(raw_payload),
        "min_timestamp": str(actual_frame.index[0]),
        "max_timestamp": str(actual_frame.index[-1]),
        "fingerprint_schema": "equity-market-data-v2",
        "data_fingerprint": dataframe_fingerprint(actual_frame, market_manifest),
        "validation": None,
        "status": "COMPLETE",
        "live_orders_called": False,
        "manifest_fingerprint": _digest("chunk-manifest-v1"),
    }
    immutable_chunk = {
        key: value
        for key, value in chunk.items()
        if key
        not in {
            "retrieval_timestamp",
            "validation",
            "error",
            "status",
            "manifest_fingerprint",
            "raw_path",
        }
    }
    parent_raw_sha = raw_sha256 or aggregate_raw_sha([raw_digest])
    manifest_identity = {
        "schema_version": 2,
        "artifact_type": "upstox_v3_historical_dataset",
        "request": request,
        "pit_evidence_fingerprint": pit,
        "corporate_action_evidence_fingerprint": corporate_actions,
        "acquisition_plan_fingerprint": plan,
        "session_policy_identity": session_identity,
        "evidence_fingerprint": _digest("evidence-v1"),
        "adjustment_policy": adjustment_policy,
        "universe_rule_version": "synthetic-pit-v1",
        "raw_sha256": parent_raw_sha,
        "raw_artifacts_identity": [immutable_chunk],
        "requested_dates": [day.isoformat() for day in trade_dates],
        "status": status.value,
    }
    manifest_payload = {
        **manifest_identity,
        "retrieval_timestamp": datetime(2026, 9, 10, tzinfo=UTC).isoformat(),
        "requested_start": trade_dates[0].isoformat(),
        "requested_end": trade_dates[-1].isoformat(),
        "covered_dates": [day.isoformat() for day in trade_dates],
        "raw_bytes": len(raw_payload),
        "rows": len(actual_frame),
        "timezone": timezone,
        "min_timestamp": str(actual_frame.index[0]),
        "max_timestamp": str(actual_frame.index[-1]),
        "fingerprint_schema": "equity-market-data-v2",
        "failure": None,
        "raw_artifacts": [chunk],
        "manifest_fingerprint": canonical_sha256(manifest_identity),
        "live_orders_called": False,
        "parquet_path": str(parquet_path),
        "parquet_sha256": hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
        "manifest_path": str(tmp_path / "synthetic.manifest.json"),
        "market_data_manifest": asdict(market_manifest),
        "validation": None,
        "continuous_session_rows": 2,
        "cas_auxiliary_rows": 2,
    }
    manifest = AcquisitionArtifactManifest.from_kiro_payload(manifest_payload)
    return RawAcquisitionArtifact(
        raw_path=parquet_path,
        manifest=manifest,
        market_data_manifest=market_manifest,
        raw_byte_paths=(raw_path,),
        parquet_sha256=hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
    )


def _run(artifact: RawAcquisitionArtifact):
    return build_validated_dataset_handoff(
        artifact,
        expected_instrument_key=artifact.manifest.instrument_key,
        expected_interval=artifact.manifest.interval,
        session_policy=NSEEquitySessionPolicy(cas_eligible=True, exit_buffer_minutes=0),
    )


def test_complete_artifact_becomes_continuous_session_research_input(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    verified, descriptor, research_input = _run(artifact)
    _, repeat_descriptor, repeat_input = _run(artifact)

    assert verified.live_orders_called is False
    assert descriptor.live_orders_called is False
    assert research_input.live_orders_called is False
    assert descriptor.raw_row_count == 4
    assert descriptor.research_row_count == 2
    assert descriptor.excluded_auxiliary_row_count == 2
    assert list(research_input.frame.index.time) == [
        pd.Timestamp("09:15").time(),
        pd.Timestamp("15:10").time(),
    ]
    assert all(
        timestamp.time() < pd.Timestamp("15:15").time() for timestamp in research_input.frame.index
    )
    assert len(descriptor.deterministic_fingerprint()) == 64
    assert repeat_descriptor.deterministic_fingerprint() == descriptor.deterministic_fingerprint()
    assert repeat_input.fingerprint == research_input.fingerprint


def test_descriptor_claim_not_bound_to_verified_acquisition_fails_closed(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    verified, descriptor, _ = _run(artifact)
    forged = replace(descriptor, pit_fingerprint=_digest("forged-pit"))
    with pytest.raises(HandoffValidationError, match="not bound"):
        build_continuous_session_research_input(
            verified,
            forged,
            session_policy=NSEEquitySessionPolicy(cas_eligible=True, exit_buffer_minutes=0),
        )


def test_research_input_integrity_succeeds_before_consumption(tmp_path: Path) -> None:
    _, descriptor, research_input = _run(_artifact(tmp_path))

    assert research_input.validate_integrity() is None
    assert research_input.fingerprint == descriptor.deterministic_fingerprint()
    detached = research_input.frame_for_research()
    detached.iloc[0, detached.columns.get_loc("close")] += 1
    assert research_input.validate_integrity() is None


@pytest.mark.parametrize("mutation", ["row", "timestamp", "volume", "value"])
def test_research_input_mutation_fails_integrity_and_trusted_access(
    tmp_path: Path, mutation: str
) -> None:
    artifact = _artifact(tmp_path)
    _, _, research_input = _run(artifact)
    original_raw = artifact.raw_path.read_bytes()

    if mutation == "row":
        research_input.frame.drop(index=research_input.frame.index[0], inplace=True)
    elif mutation == "timestamp":
        changed_index = list(research_input.frame.index)
        changed_index[0] = pd.Timestamp("2026-09-07 09:20", tz="Asia/Kolkata")
        research_input.frame.index = pd.DatetimeIndex(changed_index)
    elif mutation == "volume":
        research_input.frame.iloc[0, research_input.frame.columns.get_loc("volume")] += 1
    else:
        research_input.frame.iloc[0, research_input.frame.columns.get_loc("close")] += 1

    with pytest.raises(HandoffValidationError, match="changed after validation"):
        research_input.validate_integrity()
    with pytest.raises(HandoffValidationError, match="changed after validation"):
        _ = research_input.fingerprint
    with pytest.raises(HandoffValidationError, match="changed after validation"):
        research_input.frame_for_research()
    assert artifact.raw_path.read_bytes() == original_raw


@pytest.mark.parametrize("status", [AcquisitionStatus.PARTIAL, AcquisitionStatus.FAILED])
def test_non_complete_artifacts_fail_closed(tmp_path: Path, status: AcquisitionStatus) -> None:
    with pytest.raises(HandoffValidationError, match="not research-ready"):
        _run(_artifact(tmp_path, status=status))


def test_raw_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(HandoffValidationError, match="SHA-256"):
        _run(_artifact(tmp_path, raw_sha256=_digest("wrong-raw")))


def test_instrument_and_interval_mismatch_fail_closed(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    with pytest.raises(HandoffValidationError, match="instrument"):
        build_validated_dataset_handoff(
            artifact,
            expected_instrument_key="NSE_EQ|OTHER",
            expected_interval="5m",
            session_policy=NSEEquitySessionPolicy(cas_eligible=True, exit_buffer_minutes=0),
        )
    with pytest.raises(HandoffValidationError, match="interval"):
        build_validated_dataset_handoff(
            artifact,
            expected_instrument_key=artifact.manifest.instrument_key,
            expected_interval="1m",
            session_policy=NSEEquitySessionPolicy(cas_eligible=True, exit_buffer_minutes=0),
        )


def test_timezone_mismatch_fails_closed(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, timezone="UTC")
    with pytest.raises(HandoffValidationError, match="timezone"):
        _run(artifact)


def test_duplicate_timestamps_fail_closed(tmp_path: Path) -> None:
    frame = _frame()
    duplicate = pd.concat([frame, frame.iloc[[1]]]).sort_index()
    with pytest.raises(HandoffValidationError, match="duplicate"):
        _run(_artifact(tmp_path, frame=duplicate))


def test_missing_evidence_never_defaults(tmp_path: Path) -> None:
    with pytest.raises(HandoffValidationError, match="pit.*fingerprint"):
        _artifact(tmp_path, pit_fingerprint="unknown")


def test_pit_corporate_action_and_session_mutations_change_identity(tmp_path: Path) -> None:
    first = _artifact(tmp_path)
    _, first_descriptor, _ = _run(first)

    pit_changed = _artifact(tmp_path, pit_fingerprint=_digest("pit-v2"))
    _, pit_descriptor, _ = _run(pit_changed)
    assert (
        pit_descriptor.deterministic_fingerprint() != first_descriptor.deterministic_fingerprint()
    )

    ca_changed = _artifact(tmp_path, corporate_action_fingerprint=_digest("ca-v2"))
    _, ca_descriptor, _ = _run(ca_changed)
    assert ca_descriptor.deterministic_fingerprint() != first_descriptor.deterministic_fingerprint()

    changed_policy = NSEEquitySessionPolicy(cas_eligible=True, exit_buffer_minutes=5)
    policy_changed = _artifact(tmp_path, session_policy=changed_policy)
    changed_result = build_validated_dataset_handoff(
        policy_changed,
        expected_instrument_key=policy_changed.manifest.instrument_key,
        expected_interval=policy_changed.manifest.interval,
        session_policy=changed_policy,
    )
    assert (
        changed_result[1].deterministic_fingerprint()
        != first_descriptor.deterministic_fingerprint()
    )


def test_adjustment_policy_mutation_changes_identity(tmp_path: Path) -> None:
    first = _artifact(tmp_path)
    _, first_descriptor, _ = _run(first)
    changed = _artifact(tmp_path, adjustment_policy="split-adjusted")
    _, changed_descriptor, _ = _run(changed)
    assert (
        changed_descriptor.deterministic_fingerprint()
        != first_descriptor.deterministic_fingerprint()
    )


def test_raw_mutation_is_detected_and_raw_file_is_never_rewritten(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    _run(artifact)
    original_bytes = artifact.raw_path.read_bytes()
    changed = pd.read_parquet(artifact.raw_path)
    changed.iloc[0, changed.columns.get_loc("close")] += 1
    changed.to_parquet(artifact.raw_path)
    assert artifact.raw_path.read_bytes() != original_bytes
    with pytest.raises(HandoffValidationError, match="SHA-256"):
        verify_raw_acquisition(
            artifact,
            expected_instrument_key=artifact.manifest.instrument_key,
            expected_interval=artifact.manifest.interval,
        )
