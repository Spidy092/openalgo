from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from equity_engine.market_sessions import NSEEquitySessionPolicy
from equity_engine.provenance import MarketDataManifest
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
    raw_path = tmp_path / "raw.parquet"
    actual_frame.to_parquet(raw_path)
    raw_digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
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
    manifest = AcquisitionArtifactManifest(
        instrument_key=instrument_key,
        interval=interval,
        requested_start=trade_dates[0],
        requested_end=trade_dates[-1],
        requested_dates=trade_dates,
        covered_dates=trade_dates,
        raw_sha256=raw_sha256 or raw_digest,
        status=status,
        timezone=timezone,
        row_count=len(actual_frame),
        pit_fingerprint=pit_fingerprint or _digest("pit-v1"),
        corporate_action_fingerprint=corporate_action_fingerprint or _digest("ca-v1"),
        acquisition_plan_fingerprint=_digest("plan-v1"),
        session_policy_identity=session_policy_fingerprint(policy),
        adjustment_policy=adjustment_policy,
        source_reference="synthetic:local-fixture",
    )
    return RawAcquisitionArtifact(
        raw_path=raw_path,
        manifest=manifest,
        market_data_manifest=market_manifest,
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
    with pytest.raises(HandoffValidationError, match="pit_fingerprint"):
        _artifact(tmp_path, pit_fingerprint="unknown")


def test_pit_corporate_action_and_session_mutations_change_identity(tmp_path: Path) -> None:
    first = _artifact(tmp_path)
    _, first_descriptor, _ = _run(first)

    pit_changed = replace(
        first, manifest=replace(first.manifest, pit_fingerprint=_digest("pit-v2"))
    )
    _, pit_descriptor, _ = _run(pit_changed)
    assert (
        pit_descriptor.deterministic_fingerprint() != first_descriptor.deterministic_fingerprint()
    )

    ca_changed = replace(
        first,
        manifest=replace(first.manifest, corporate_action_fingerprint=_digest("ca-v2")),
    )
    _, ca_descriptor, _ = _run(ca_changed)
    assert ca_descriptor.deterministic_fingerprint() != first_descriptor.deterministic_fingerprint()

    changed_policy = NSEEquitySessionPolicy(cas_eligible=True, exit_buffer_minutes=5)
    policy_changed = replace(
        first,
        manifest=replace(
            first.manifest,
            session_policy_identity=session_policy_fingerprint(changed_policy),
        ),
    )
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
    changed = replace(first, manifest=replace(first.manifest, adjustment_policy="split-adjusted"))
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
