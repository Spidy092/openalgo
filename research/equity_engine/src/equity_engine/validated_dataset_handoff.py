"""Read-only handoff from a verified acquisition artifact to research data.

This module is deliberately a boundary, not an acquisition system.  It consumes a local raw
artifact and the existing :class:`MarketDataManifest`, verifies their binding, and produces an
immutable provenance descriptor plus a copy of the continuous-session bars used by research.
There are no network calls, credentials, downloads, or writes in this module.

The PIT, corporate-action, acquisition-plan, session, and adjustment fields are fingerprints or
identities supplied by their owning subsystems.  This boundary binds them; it does not create a
second implementation of any of those systems and never treats missing/unknown evidence as zero.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, time
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd

from .market_sessions import ContinuousSessionPolicy, filter_to_continuous_session
from .provenance import (
    FINGERPRINT_SCHEMA,
    MarketDataManifest,
    dataframe_fingerprint,
    validate_ohlcv_frame,
)

HANDOFF_SCHEMA_VERSION = "validated-dataset-handoff/v1"
_HEX_DIGEST_LENGTH = 64


class HandoffValidationError(ValueError):
    """Raised when a raw acquisition cannot be admitted to research."""


class AcquisitionStatus(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


def _canonical_value(value: Any) -> Any:
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    return value


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        _canonical_value(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_digest(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _HEX_DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HandoffValidationError(f"{name} must be a 64-character lowercase SHA-256 digest")
    return value


def _require_nonempty(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value.strip().lower() in {"unknown", "missing", "none", "null"}
    ):
        raise HandoffValidationError(f"{name} is required and cannot be unknown")
    return value


def _date_tuple(name: str, values: tuple[date, ...]) -> tuple[date, ...]:
    if not values:
        raise HandoffValidationError(f"{name} must contain at least one date")
    if tuple(sorted(set(values))) != values:
        raise HandoffValidationError(f"{name} must be sorted and unique")
    return values


@dataclass(frozen=True)
class AcquisitionArtifactManifest:
    """The future Kiro-to-research manifest contract.

    ``requested_dates`` are the exact dates requested from the acquisition plan, not an inferred
    weekday range.  A COMPLETE artifact must cover exactly those dates.  This keeps holidays and
    special sessions explicit instead of silently treating absent dates as missing candles.
    """

    instrument_key: str
    interval: str
    requested_start: date
    requested_end: date
    requested_dates: tuple[date, ...]
    covered_dates: tuple[date, ...]
    raw_sha256: str
    status: AcquisitionStatus
    timezone: str
    row_count: int
    pit_fingerprint: str
    corporate_action_fingerprint: str
    acquisition_plan_fingerprint: str
    session_policy_identity: str
    adjustment_policy: str
    source_reference: str
    raw_format: str = "parquet"

    def __post_init__(self) -> None:
        _require_nonempty("instrument_key", self.instrument_key)
        _require_nonempty("interval", self.interval)
        _require_nonempty("timezone", self.timezone)
        _require_nonempty("session_policy_identity", self.session_policy_identity)
        _require_nonempty("adjustment_policy", self.adjustment_policy)
        _require_nonempty("source_reference", self.source_reference)
        if self.requested_start > self.requested_end:
            raise HandoffValidationError("requested date boundaries are invalid")
        _date_tuple("requested_dates", self.requested_dates)
        _date_tuple("covered_dates", self.covered_dates)
        if self.requested_dates[0] < self.requested_start:
            raise HandoffValidationError("requested_dates precede requested_start")
        if self.requested_dates[-1] > self.requested_end:
            raise HandoffValidationError("requested_dates exceed requested_end")
        if any(
            day < self.requested_start or day > self.requested_end for day in self.covered_dates
        ):
            raise HandoffValidationError("covered_dates exceed requested date boundaries")
        if not set(self.covered_dates).issubset(self.requested_dates):
            raise HandoffValidationError("covered_dates contain dates that were not requested")
        _require_digest("raw_sha256", self.raw_sha256)
        for name, value in (
            ("pit_fingerprint", self.pit_fingerprint),
            ("corporate_action_fingerprint", self.corporate_action_fingerprint),
            ("acquisition_plan_fingerprint", self.acquisition_plan_fingerprint),
        ):
            _require_digest(name, value)
        if not isinstance(self.status, AcquisitionStatus):
            raise HandoffValidationError("status must be COMPLETE, PARTIAL, or FAILED")
        if (
            not isinstance(self.row_count, int)
            or isinstance(self.row_count, bool)
            or self.row_count <= 0
        ):
            raise HandoffValidationError("row_count must be a positive integer")
        if self.raw_format != "parquet":
            raise HandoffValidationError("only local parquet acquisition artifacts are supported")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HANDOFF_SCHEMA_VERSION,
            "instrument_key": self.instrument_key,
            "interval": self.interval,
            "requested_start": self.requested_start.isoformat(),
            "requested_end": self.requested_end.isoformat(),
            "requested_dates": [day.isoformat() for day in self.requested_dates],
            "covered_dates": [day.isoformat() for day in self.covered_dates],
            "raw_sha256": self.raw_sha256,
            "status": self.status.value,
            "timezone": self.timezone,
            "row_count": self.row_count,
            "pit_fingerprint": self.pit_fingerprint,
            "corporate_action_fingerprint": self.corporate_action_fingerprint,
            "acquisition_plan_fingerprint": self.acquisition_plan_fingerprint,
            "session_policy_identity": self.session_policy_identity,
            "adjustment_policy": self.adjustment_policy,
            "source_reference": self.source_reference,
            "raw_format": self.raw_format,
        }

    def fingerprint(self) -> str:
        return _canonical_sha256(self.as_dict())


@dataclass(frozen=True)
class RawAcquisitionArtifact:
    """A local immutable-by-contract raw file plus its acquisition manifest."""

    raw_path: Path
    manifest: AcquisitionArtifactManifest
    market_data_manifest: MarketDataManifest

    def __post_init__(self) -> None:
        if not isinstance(self.raw_path, Path):
            raise HandoffValidationError("raw_path must be a pathlib.Path")


@dataclass(frozen=True)
class VerifiedRawAcquisition:
    """Verified raw frame; the frame is never written back to its source path."""

    frame: pd.DataFrame
    artifact: RawAcquisitionArtifact
    manifest_fingerprint: str
    raw_data_fingerprint: str

    @property
    def live_orders_called(self) -> bool:
        return False


@dataclass(frozen=True)
class ValidatedDatasetDescriptor:
    """Immutable identity for the research-ready continuous-session dataset."""

    schema_version: str
    instrument_key: str
    interval: str
    requested_start: date
    requested_end: date
    requested_dates: tuple[date, ...]
    covered_dates: tuple[date, ...]
    raw_sha256: str
    raw_row_count: int
    research_row_count: int
    excluded_auxiliary_row_count: int
    timezone: str
    pit_fingerprint: str
    corporate_action_fingerprint: str
    acquisition_plan_fingerprint: str
    session_policy_identity: str
    adjustment_policy: str
    source_reference: str
    acquisition_manifest_fingerprint: str
    raw_data_fingerprint: str
    research_data_fingerprint: str
    fingerprint_schema: str = FINGERPRINT_SCHEMA
    live_orders_called: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != HANDOFF_SCHEMA_VERSION:
            raise HandoffValidationError("unsupported validated dataset descriptor schema")
        if self.live_orders_called:
            raise HandoffValidationError("live orders are forbidden in the research handoff")
        if self.research_row_count <= 0 or self.raw_row_count <= 0:
            raise HandoffValidationError("validated dataset row counts must be positive")
        if self.excluded_auxiliary_row_count < 0:
            raise HandoffValidationError("excluded_auxiliary_row_count cannot be negative")
        if self.raw_row_count != self.research_row_count + self.excluded_auxiliary_row_count:
            raise HandoffValidationError("raw/research row counts do not reconcile")
        _require_digest("raw_sha256", self.raw_sha256)
        for name, value in (
            ("acquisition_manifest_fingerprint", self.acquisition_manifest_fingerprint),
            ("raw_data_fingerprint", self.raw_data_fingerprint),
            ("research_data_fingerprint", self.research_data_fingerprint),
            ("pit_fingerprint", self.pit_fingerprint),
            ("corporate_action_fingerprint", self.corporate_action_fingerprint),
            ("acquisition_plan_fingerprint", self.acquisition_plan_fingerprint),
        ):
            _require_digest(name, value)
        _require_nonempty("instrument_key", self.instrument_key)
        _require_nonempty("interval", self.interval)
        _require_nonempty("timezone", self.timezone)
        _require_nonempty("session_policy_identity", self.session_policy_identity)
        _require_nonempty("adjustment_policy", self.adjustment_policy)
        _require_nonempty("source_reference", self.source_reference)
        if self.requested_start > self.requested_end:
            raise HandoffValidationError("descriptor date boundaries are invalid")
        _date_tuple("descriptor requested_dates", self.requested_dates)
        _date_tuple("descriptor covered_dates", self.covered_dates)
        if self.requested_dates[0] < self.requested_start:
            raise HandoffValidationError("descriptor requested_dates precede requested_start")
        if self.requested_dates[-1] > self.requested_end:
            raise HandoffValidationError("descriptor requested_dates exceed requested_end")
        if self.covered_dates != self.requested_dates:
            raise HandoffValidationError(
                "research-ready descriptor must cover every requested date"
            )
        if self.fingerprint_schema != FINGERPRINT_SCHEMA:
            raise HandoffValidationError("unsupported deterministic data fingerprint schema")

    def deterministic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "instrument_key": self.instrument_key,
            "interval": self.interval,
            "requested_start": self.requested_start.isoformat(),
            "requested_end": self.requested_end.isoformat(),
            "requested_dates": [day.isoformat() for day in self.requested_dates],
            "covered_dates": [day.isoformat() for day in self.covered_dates],
            "raw_sha256": self.raw_sha256,
            "raw_row_count": self.raw_row_count,
            "research_row_count": self.research_row_count,
            "excluded_auxiliary_row_count": self.excluded_auxiliary_row_count,
            "timezone": self.timezone,
            "pit_fingerprint": self.pit_fingerprint,
            "corporate_action_fingerprint": self.corporate_action_fingerprint,
            "acquisition_plan_fingerprint": self.acquisition_plan_fingerprint,
            "session_policy_identity": self.session_policy_identity,
            "adjustment_policy": self.adjustment_policy,
            "source_reference": self.source_reference,
            "acquisition_manifest_fingerprint": self.acquisition_manifest_fingerprint,
            "raw_data_fingerprint": self.raw_data_fingerprint,
            "research_data_fingerprint": self.research_data_fingerprint,
            "fingerprint_schema": self.fingerprint_schema,
            "live_orders_called": False,
        }

    def deterministic_fingerprint(self) -> str:
        return _canonical_sha256(self.deterministic_payload())

    def as_dict(self) -> dict[str, Any]:
        payload = self.deterministic_payload()
        payload["descriptor_fingerprint"] = self.deterministic_fingerprint()
        return payload


@dataclass(frozen=True)
class ContinuousSessionResearchInput:
    """Research-only frame after effective session filtering."""

    frame: pd.DataFrame
    descriptor: ValidatedDatasetDescriptor
    market_data_manifest: MarketDataManifest

    def validate_integrity(self) -> None:
        """Fail closed if the in-memory research frame changed after validation."""

        try:
            current_fingerprint = dataframe_fingerprint(self.frame, self.market_data_manifest)
        except (TypeError, ValueError) as exc:
            raise HandoffValidationError(
                "in-memory research frame is no longer fingerprintable"
            ) from exc
        if current_fingerprint != self.descriptor.research_data_fingerprint:
            raise HandoffValidationError("in-memory research frame changed after validation")

    def frame_for_research(self) -> pd.DataFrame:
        """Return a detached research copy only after integrity verification."""

        self.validate_integrity()
        return self.frame.copy(deep=True)

    @property
    def fingerprint(self) -> str:
        self.validate_integrity()
        return self.descriptor.deterministic_fingerprint()

    @property
    def live_orders_called(self) -> bool:
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise HandoffValidationError(f"cannot read raw acquisition artifact: {path}") from exc
    return digest.hexdigest()


def session_policy_fingerprint(policy: ContinuousSessionPolicy) -> str:
    """Derive a stable identity from a dataclass-backed effective session policy."""

    if not dataclasses.is_dataclass(policy):
        raise HandoffValidationError(
            "session policy must expose dataclass state so its identity is auditable"
        )
    payload = {
        "class": f"{type(policy).__module__}.{type(policy).__qualname__}",
        "state": dataclasses.asdict(policy),
    }
    return _canonical_sha256(payload)


def _assert_verified_frame_unchanged(verified: VerifiedRawAcquisition) -> None:
    """Reject mutation of either the source file or the in-memory verified frame."""

    artifact = verified.artifact
    if _sha256_file(artifact.raw_path) != artifact.manifest.raw_sha256:
        raise HandoffValidationError("raw acquisition SHA-256 no longer matches its manifest")
    frame = verified.frame
    if len(frame) != artifact.manifest.row_count:
        raise HandoffValidationError("verified frame row count no longer matches its manifest")
    try:
        current_fingerprint = dataframe_fingerprint(frame, artifact.market_data_manifest)
    except (TypeError, ValueError) as exc:
        raise HandoffValidationError("verified frame is no longer fingerprintable") from exc
    if current_fingerprint != verified.raw_data_fingerprint:
        raise HandoffValidationError("verified in-memory frame changed after raw verification")


def verify_raw_acquisition(
    artifact: RawAcquisitionArtifact,
    *,
    expected_instrument_key: str,
    expected_interval: str,
) -> VerifiedRawAcquisition:
    """Verify a COMPLETE local artifact without mutating or writing the raw file."""

    manifest = artifact.manifest
    if manifest.status is not AcquisitionStatus.COMPLETE:
        raise HandoffValidationError(
            f"acquisition status {manifest.status.value} is not research-ready"
        )
    if manifest.instrument_key != expected_instrument_key:
        raise HandoffValidationError(
            "acquisition instrument does not match the requested instrument"
        )
    if manifest.interval != expected_interval:
        raise HandoffValidationError("acquisition interval does not match the requested interval")
    market_manifest = artifact.market_data_manifest
    if market_manifest.instrument_token != manifest.instrument_key:
        raise HandoffValidationError("market-data manifest instrument does not match acquisition")
    if market_manifest.interval != manifest.interval:
        raise HandoffValidationError("market-data manifest interval does not match acquisition")
    if market_manifest.timezone != manifest.timezone:
        raise HandoffValidationError("market-data manifest timezone does not match acquisition")

    if not artifact.raw_path.is_file():
        raise HandoffValidationError("raw acquisition artifact does not exist")
    if _sha256_file(artifact.raw_path) != manifest.raw_sha256:
        raise HandoffValidationError("raw acquisition SHA-256 does not match its manifest")
    try:
        frame = pd.read_parquet(artifact.raw_path)
    except Exception as exc:
        raise HandoffValidationError("raw acquisition artifact is not readable parquet") from exc
    if len(frame) != manifest.row_count:
        raise HandoffValidationError("raw row count does not match its manifest")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise HandoffValidationError("raw acquisition index must be a DatetimeIndex")
    if frame.index.tz is None or str(frame.index.tz) != manifest.timezone:
        raise HandoffValidationError("raw acquisition timezone does not match its manifest")
    violations = validate_ohlcv_frame(frame)
    if violations:
        raise HandoffValidationError(
            "raw acquisition failed structural validation: " + "; ".join(violations)
        )
    actual_dates = tuple(sorted(set(frame.index.date)))
    if actual_dates != manifest.covered_dates:
        raise HandoffValidationError("raw covered dates do not match its manifest")
    if manifest.covered_dates != manifest.requested_dates:
        raise HandoffValidationError("COMPLETE acquisition does not cover every requested date")
    if frame.index[0].date() < manifest.requested_start:
        raise HandoffValidationError("raw timestamps precede requested_start")
    if frame.index[-1].date() > manifest.requested_end:
        raise HandoffValidationError("raw timestamps exceed requested_end")
    try:
        raw_data_fingerprint = dataframe_fingerprint(frame, market_manifest)
    except (TypeError, ValueError) as exc:
        raise HandoffValidationError(
            "raw acquisition cannot receive a deterministic fingerprint"
        ) from exc
    return VerifiedRawAcquisition(
        frame=frame.copy(deep=True),
        artifact=artifact,
        manifest_fingerprint=manifest.fingerprint(),
        raw_data_fingerprint=raw_data_fingerprint,
    )


def _continuous_frame(
    verified: VerifiedRawAcquisition,
    session_policy: ContinuousSessionPolicy,
    *,
    expected_policy_identity: str,
) -> tuple[pd.DataFrame, int, str]:
    _assert_verified_frame_unchanged(verified)
    if session_policy_fingerprint(session_policy) != expected_policy_identity:
        raise HandoffValidationError("runtime session policy does not match bound session identity")
    continuous = filter_to_continuous_session(verified.frame, session_policy)
    if continuous.empty:
        raise HandoffValidationError("continuous-session research input is empty")
    continuous_dates = tuple(sorted(set(continuous.index.date)))
    if continuous_dates != verified.artifact.manifest.covered_dates:
        raise HandoffValidationError(
            "at least one covered date has no continuous-session research bars"
        )
    violations = validate_ohlcv_frame(continuous)
    if violations:
        raise HandoffValidationError(
            "continuous-session research input failed validation: " + "; ".join(violations)
        )
    continuous_data_fingerprint = dataframe_fingerprint(
        continuous, verified.artifact.market_data_manifest
    )
    return continuous, len(verified.frame) - len(continuous), continuous_data_fingerprint


def build_validated_dataset_descriptor(
    verified: VerifiedRawAcquisition,
    *,
    session_policy: ContinuousSessionPolicy,
) -> ValidatedDatasetDescriptor:
    """Build the descriptor only after raw and continuous-session validation succeeds."""

    manifest = verified.artifact.manifest
    continuous, excluded, continuous_data_fingerprint = _continuous_frame(
        verified,
        session_policy,
        expected_policy_identity=manifest.session_policy_identity,
    )
    descriptor = ValidatedDatasetDescriptor(
        schema_version=HANDOFF_SCHEMA_VERSION,
        instrument_key=manifest.instrument_key,
        interval=manifest.interval,
        requested_start=manifest.requested_start,
        requested_end=manifest.requested_end,
        requested_dates=manifest.requested_dates,
        covered_dates=manifest.covered_dates,
        raw_sha256=manifest.raw_sha256,
        raw_row_count=len(verified.frame),
        research_row_count=len(continuous),
        excluded_auxiliary_row_count=excluded,
        timezone=manifest.timezone,
        pit_fingerprint=manifest.pit_fingerprint,
        corporate_action_fingerprint=manifest.corporate_action_fingerprint,
        acquisition_plan_fingerprint=manifest.acquisition_plan_fingerprint,
        session_policy_identity=manifest.session_policy_identity,
        adjustment_policy=manifest.adjustment_policy,
        source_reference=manifest.source_reference,
        acquisition_manifest_fingerprint=verified.manifest_fingerprint,
        raw_data_fingerprint=verified.raw_data_fingerprint,
        research_data_fingerprint=continuous_data_fingerprint,
    )
    return descriptor


def _assert_descriptor_matches_verified(
    verified: VerifiedRawAcquisition,
    descriptor: ValidatedDatasetDescriptor,
) -> None:
    manifest = verified.artifact.manifest
    expected = {
        "instrument_key": manifest.instrument_key,
        "interval": manifest.interval,
        "requested_start": manifest.requested_start,
        "requested_end": manifest.requested_end,
        "requested_dates": manifest.requested_dates,
        "covered_dates": manifest.covered_dates,
        "raw_sha256": manifest.raw_sha256,
        "raw_row_count": len(verified.frame),
        "timezone": manifest.timezone,
        "pit_fingerprint": manifest.pit_fingerprint,
        "corporate_action_fingerprint": manifest.corporate_action_fingerprint,
        "acquisition_plan_fingerprint": manifest.acquisition_plan_fingerprint,
        "session_policy_identity": manifest.session_policy_identity,
        "adjustment_policy": manifest.adjustment_policy,
        "source_reference": manifest.source_reference,
        "acquisition_manifest_fingerprint": verified.manifest_fingerprint,
        "raw_data_fingerprint": verified.raw_data_fingerprint,
    }
    mismatches = [
        name
        for name, expected_value in expected.items()
        if getattr(descriptor, name) != expected_value
    ]
    if mismatches:
        raise HandoffValidationError(
            "descriptor is not bound to the verified acquisition: " + ", ".join(mismatches)
        )


def build_continuous_session_research_input(
    verified: VerifiedRawAcquisition,
    descriptor: ValidatedDatasetDescriptor,
    *,
    session_policy: ContinuousSessionPolicy,
) -> ContinuousSessionResearchInput:
    """Return a research-only copy and verify it still matches the descriptor."""

    _assert_descriptor_matches_verified(verified, descriptor)
    if descriptor.deterministic_fingerprint() != descriptor.as_dict()["descriptor_fingerprint"]:
        raise HandoffValidationError("descriptor deterministic fingerprint is not self-consistent")
    continuous, excluded, continuous_data_fingerprint = _continuous_frame(
        verified,
        session_policy,
        expected_policy_identity=descriptor.session_policy_identity,
    )
    if excluded != descriptor.excluded_auxiliary_row_count:
        raise HandoffValidationError("descriptor auxiliary-row count does not match input")
    if continuous_data_fingerprint != descriptor.research_data_fingerprint:
        raise HandoffValidationError("descriptor research fingerprint does not match input")
    return ContinuousSessionResearchInput(
        frame=continuous.copy(deep=True),
        descriptor=descriptor,
        market_data_manifest=verified.artifact.market_data_manifest,
    )


def build_validated_dataset_handoff(
    artifact: RawAcquisitionArtifact,
    *,
    expected_instrument_key: str,
    expected_interval: str,
    session_policy: ContinuousSessionPolicy,
) -> tuple[VerifiedRawAcquisition, ValidatedDatasetDescriptor, ContinuousSessionResearchInput]:
    """Run the complete local handoff in one explicit, read-only operation."""

    verified = verify_raw_acquisition(
        artifact,
        expected_instrument_key=expected_instrument_key,
        expected_interval=expected_interval,
    )
    descriptor = build_validated_dataset_descriptor(verified, session_policy=session_policy)
    research_input = build_continuous_session_research_input(
        verified,
        descriptor,
        session_policy=session_policy,
    )
    return verified, descriptor, research_input
