"""Canonical Experiment V3 API with hardened PIT corporate-action trust validation.

The complete pre-existing Experiment V3 implementation is preserved verbatim in
``_experiment_v3_core``.  This module extends that canonical artifact at the corporate-action
trust boundary without replacing the effective-dated cost ledger, current-calibration guard,
provenance, promotion evidence, or live-order prohibition already present in V3.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from . import _experiment_v3_core as _core
from ._experiment_v3_core import *  # noqa: F401,F403
from .corporate_actions import CorporateActionEvaluationMode
from .gates import PromotionThresholds
from .provenance import MarketDataManifest


class CorporateActionMismatchError(MissingEvidenceError):
    """Raised when a corporate-action evidence claim disagrees with trusted evidence."""


@dataclass(frozen=True)
class CorporateActionEvidenceIdentity:
    """Serialized corporate-action evidence claim that must be revalidated before trust.

    ``authoritative`` is a claim carried by the artifact, not proof.  Promotion and integrity
    validation always recompute the expected identity from a trusted PIT ledger and compare the
    complete evidence boundary, including exact dates, instruments, policy, events, and source.
    """

    source: str
    complete: bool
    blocking_events: tuple[str, ...]
    evidence_fingerprint: str
    coverage_start: date | None = None
    coverage_end: date | None = None
    covered_instruments: tuple[str, ...] = ()
    events_count: int = 0
    policy_identity: str = "DEFAULT"
    authoritative: bool = False
    evaluation_mode: CorporateActionEvaluationMode = (
        CorporateActionEvaluationMode.TRADABLE_INFORMATION
    )

    def __post_init__(self) -> None:
        if isinstance(self.evaluation_mode, str):
            object.__setattr__(
                self,
                "evaluation_mode",
                CorporateActionEvaluationMode(self.evaluation_mode),
            )
        if not self.complete:
            raise ValueError("corporate-action evidence must be complete")
        if not self.evidence_fingerprint.strip():
            raise ValueError("corporate-action evidence fingerprint is required")
        if not self.covered_instruments:
            raise ValueError("covered_instruments cannot be empty for corporate-action evidence")
        if any(not str(key).strip() for key in self.covered_instruments):
            raise ValueError("covered_instruments cannot contain empty instrument keys")
        if len(set(self.covered_instruments)) != len(self.covered_instruments):
            raise ValueError("covered_instruments must be unique; duplicates are strictly forbidden")
        if tuple(sorted(self.covered_instruments)) != tuple(self.covered_instruments):
            raise ValueError("covered_instruments must be in sorted canonical order")
        if self.coverage_start is not None and self.coverage_end is not None:
            if self.coverage_start > self.coverage_end:
                raise ValueError("coverage_start must be on or before coverage_end")

    @property
    def is_authoritative(self) -> bool:
        """Return the serialized authority claim; callers must still revalidate it."""
        return self.authoritative

    def covers_window(self, start: date, end: date) -> bool:
        if self.coverage_start is None or self.coverage_end is None:
            return False
        return self.coverage_start <= start and self.coverage_end >= end

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "complete": self.complete,
            "blocking_events": sorted(self.blocking_events),
            "evidence_fingerprint": self.evidence_fingerprint,
            "coverage_start": self.coverage_start.isoformat() if self.coverage_start else None,
            "coverage_end": self.coverage_end.isoformat() if self.coverage_end else None,
            "covered_instruments": list(self.covered_instruments),
            "events_count": self.events_count,
            "policy_identity": self.policy_identity,
            "authoritative": self.authoritative,
            "evaluation_mode": self.evaluation_mode.value,
        }

    def identity_fingerprint(self) -> str:
        canonical_json = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def fingerprint(self) -> str:
        return self.identity_fingerprint()

    def _derive_from_trusted_ledger(
        self,
        ledger: Any,
        *,
        research_start: date | None = None,
        research_end: date | None = None,
        canonical_instruments: Iterable[str] | None = None,
        policy: Any = None,
    ) -> CorporateActionEvidenceIdentity:
        start = research_start if research_start is not None else self.coverage_start
        end = research_end if research_end is not None else self.coverage_end
        instruments = canonical_instruments if canonical_instruments is not None else self.covered_instruments
        if start is None or end is None:
            raise CorporateActionMismatchError(
                "coverage start and end dates are required to validate against trusted ledger"
            )
        if not instruments:
            raise CorporateActionMismatchError(
                "canonical instruments are required to validate against trusted ledger"
            )
        try:
            return ledger.to_evidence_identity(
                research_start=start,
                research_end=end,
                instruments=instruments,
                policy=policy,
                evaluation_mode=self.evaluation_mode,
            )
        except Exception as exc:
            raise CorporateActionMismatchError(
                f"trusted corporate-action ledger failed revalidation: {exc}"
            ) from exc

    def validate_against_trusted_ledger(
        self,
        ledger: Any,
        *,
        research_start: date | None = None,
        research_end: date | None = None,
        canonical_instruments: Iterable[str] | None = None,
        policy: Any = None,
    ) -> None:
        expected = self._derive_from_trusted_ledger(
            ledger,
            research_start=research_start,
            research_end=research_end,
            canonical_instruments=canonical_instruments,
            policy=policy,
        )
        mismatches: list[str] = []
        if self.evidence_fingerprint != expected.evidence_fingerprint:
            mismatches.append(
                f"evidence_fingerprint (expected {expected.evidence_fingerprint!r}, got {self.evidence_fingerprint!r})"
            )
        if self.coverage_start != expected.coverage_start:
            mismatches.append(
                f"coverage_start (expected {expected.coverage_start}, got {self.coverage_start})"
            )
        if research_start is not None and self.coverage_start != research_start:
            mismatches.append(
                f"research_start mismatch with coverage_start (expected {research_start}, got {self.coverage_start})"
            )
        if self.coverage_end != expected.coverage_end:
            mismatches.append(
                f"coverage_end (expected {expected.coverage_end}, got {self.coverage_end})"
            )
        if research_end is not None and self.coverage_end != research_end:
            mismatches.append(
                f"research_end mismatch with coverage_end (expected {research_end}, got {self.coverage_end})"
            )
        if tuple(sorted(self.covered_instruments)) != tuple(sorted(expected.covered_instruments)):
            mismatches.append(
                f"covered_instruments (expected {expected.covered_instruments}, got {self.covered_instruments})"
            )
        if canonical_instruments is not None:
            canonical = tuple(sorted(canonical_instruments))
            if tuple(sorted(self.covered_instruments)) != canonical:
                mismatches.append(
                    "covered_instruments vs canonical experiment instruments "
                    f"(expected {canonical}, got {tuple(sorted(self.covered_instruments))})"
                )
        if self.policy_identity != expected.policy_identity:
            mismatches.append(
                f"policy_identity (expected {expected.policy_identity!r}, got {self.policy_identity!r})"
            )
        if self.blocking_events != expected.blocking_events:
            mismatches.append(
                f"blocking_events (expected {expected.blocking_events}, got {self.blocking_events})"
            )
        if self.events_count != expected.events_count:
            mismatches.append(
                f"events_count (expected {expected.events_count}, got {self.events_count})"
            )
        if self.complete != expected.complete:
            mismatches.append(f"complete (expected {expected.complete}, got {self.complete})")
        if (
            self.coverage_start is None
            or self.coverage_end is None
            or expected.coverage_start is None
            or expected.coverage_end is None
            or not self.covers_window(expected.coverage_start, expected.coverage_end)
        ):
            mismatches.append("coverage window does not cover required window")
        if self.evaluation_mode != expected.evaluation_mode:
            mismatches.append(
                f"evaluation_mode (expected {expected.evaluation_mode.value!r}, got {self.evaluation_mode.value!r})"
            )
        if self.source != expected.source:
            mismatches.append(f"source (expected {expected.source!r}, got {self.source!r})")
        if not self.authoritative:
            mismatches.append("authoritative (claim is not marked authoritative)")
        if mismatches:
            raise CorporateActionMismatchError(
                "corporate-action evidence claim does not match trusted ledger; mismatches: "
                + "; ".join(mismatches)
            )

    @classmethod
    def from_ledger(
        cls,
        ledger: Any,
        *,
        research_window: ResearchWindowConfig | Any,
        instruments: Iterable[str],
        policy: Any = None,
        evaluation_mode: CorporateActionEvaluationMode | str = (
            CorporateActionEvaluationMode.TRADABLE_INFORMATION
        ),
        source: str | None = None,
    ) -> CorporateActionEvidenceIdentity:
        start = research_window.start if hasattr(research_window, "start") else research_window[0]
        end = research_window.end if hasattr(research_window, "end") else research_window[1]
        kwargs: dict[str, Any] = {
            "research_start": start,
            "research_end": end,
            "instruments": instruments,
            "policy": policy,
            "evaluation_mode": evaluation_mode,
        }
        if source is not None:
            kwargs["source"] = source
        return ledger.to_evidence_identity(**kwargs)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": self.source,
            "complete": self.complete,
            "blocking_events": list(self.blocking_events),
            "evidence_fingerprint": self.evidence_fingerprint,
            "authoritative": self.authoritative,
            "evaluation_mode": self.evaluation_mode.value,
        }
        if self.coverage_start is not None:
            payload["coverage_start"] = self.coverage_start.isoformat()
        if self.coverage_end is not None:
            payload["coverage_end"] = self.coverage_end.isoformat()
        if self.covered_instruments:
            payload["covered_instruments"] = list(self.covered_instruments)
        if self.events_count:
            payload["events_count"] = self.events_count
        if self.policy_identity != "DEFAULT":
            payload["policy_identity"] = self.policy_identity
        return payload


class ExperimentArtifact(_core.ExperimentArtifact):
    """Experiment V3 artifact with trusted PIT corporate-action validation layered in."""

    def _validate_corporate_action_structure(self) -> None:
        evidence = self.corporate_action_evidence
        if not evidence.complete:
            raise MissingEvidenceError("corporate-action evidence is incomplete")
        if (
            evidence.coverage_start is None
            or evidence.coverage_end is None
            or not evidence.covers_window(self.research_window.start, self.research_window.end)
        ):
            raise MissingEvidenceError(
                "experiment cannot claim corporate-action-complete unless evidence covers "
                f"exact research window [{self.research_window.start.isoformat()}, {self.research_window.end.isoformat()}]"
            )
        if evidence.blocking_events:
            raise MissingEvidenceError(
                "experiment has unresolved blocking corporate actions: "
                + ", ".join(evidence.blocking_events)
            )
        instruments = evidence.covered_instruments
        if not instruments:
            raise MissingEvidenceError("corporate-action evidence covered_instruments cannot be empty")
        if any(not str(key).strip() for key in instruments):
            raise MissingEvidenceError(
                "corporate-action covered_instruments cannot contain empty instrument keys"
            )
        if len(set(instruments)) != len(instruments):
            raise MissingEvidenceError(
                "corporate-action covered_instruments contains duplicate instruments"
            )
        if tuple(sorted(instruments)) != tuple(instruments):
            raise MissingEvidenceError(
                "corporate-action covered_instruments must be in sorted canonical order"
            )
        canonical = tuple(sorted(self.instrument_dataset_fingerprints.keys()))
        if tuple(instruments) != canonical:
            missing = sorted(set(canonical) - set(instruments))
            extra = sorted(set(instruments) - set(canonical))
            raise MissingEvidenceError(
                "corporate-action evidence population does not exactly match canonical experiment "
                f"instrument population: missing={missing}, extra={extra}"
            )

    def validate_structure(
        self,
        *,
        dataset_frames: Mapping[str, pd.DataFrame] | None = None,
        dataset_manifests: Mapping[str, MarketDataManifest] | None = None,
        prefilter_artifact: Mapping[str, Any] | None = None,
    ) -> None:
        # Preserve all existing V3 dataset, prefilter, leakage, promotion-artifact, cost, and
        # current-calibration checks while deliberately skipping threshold evaluation here.
        super().validate_integrity(
            dataset_frames=dataset_frames,
            dataset_manifests=dataset_manifests,
            prefilter_artifact=prefilter_artifact,
            promotion_thresholds=None,
        )
        self._validate_corporate_action_structure()

    def evaluate_promotion_gate(
        self,
        thresholds: PromotionThresholds,
        *,
        trusted_corporate_action_ledger: Any = None,
        trusted_corporate_action_policy: Any = None,
        corporate_action_ledger: Any = None,
        corporate_action_policy: Any = None,
        **kwargs: Any,
    ) -> tuple[bool, tuple[str, ...]]:
        base_passed, base_violations = super().evaluate_promotion_gate(thresholds)
        violations = list(base_violations)
        ledger = (
            trusted_corporate_action_ledger
            if trusted_corporate_action_ledger is not None
            else corporate_action_ledger
        )
        policy = (
            trusted_corporate_action_policy
            if trusted_corporate_action_policy is not None
            else corporate_action_policy
        )
        if ledger is None:
            violations.append(
                "trusted corporate-action evidence ledger is required for promotion; "
                "missing trusted ledger => fail closed; serialized corporate-action evidence is a claim, not proof"
            )
        else:
            try:
                self.corporate_action_evidence.validate_against_trusted_ledger(
                    ledger,
                    research_start=self.research_window.start,
                    research_end=self.research_window.end,
                    canonical_instruments=sorted(self.instrument_dataset_fingerprints.keys()),
                    policy=policy,
                )
            except (CorporateActionMismatchError, LookupError, ValueError) as exc:
                violations.append(f"corporate-action evidence integrity mismatch: {exc}")
        evidence = self.corporate_action_evidence
        if not evidence.authoritative:
            violations.append(
                "corporate-action evidence is non-authoritative claim; cannot promote without trusted ledger revalidation"
            )
        if not evidence.complete:
            violations.append("corporate-action evidence is incomplete; cannot promote")
        if evidence.blocking_events:
            violations.append(
                "corporate-action evidence has unresolved blocking events: "
                + ", ".join(evidence.blocking_events)
            )
        instruments = evidence.covered_instruments
        canonical = tuple(sorted(self.instrument_dataset_fingerprints.keys()))
        if (
            not instruments
            or len(set(instruments)) != len(instruments)
            or tuple(sorted(instruments)) != tuple(instruments)
        ):
            violations.append(
                "corporate-action covered_instruments must be sorted unique non-empty"
            )
        elif tuple(instruments) != canonical:
            violations.append(
                "corporate-action evidence does not cover exact experiment instrument population"
            )
        return (base_passed and not violations, tuple(violations))

    def validate_integrity(
        self,
        *,
        dataset_frames: Mapping[str, pd.DataFrame] | None = None,
        dataset_manifests: Mapping[str, MarketDataManifest] | None = None,
        prefilter_artifact: Mapping[str, Any] | None = None,
        promotion_thresholds: PromotionThresholds | None = None,
        corporate_action_ledger: Any = None,
        corporate_action_policy: Any = None,
        trusted_corporate_action_ledger: Any = None,
        trusted_corporate_action_policy: Any = None,
        **kwargs: Any,
    ) -> None:
        self.validate_structure(
            dataset_frames=dataset_frames,
            dataset_manifests=dataset_manifests,
            prefilter_artifact=prefilter_artifact,
        )
        ledger = (
            trusted_corporate_action_ledger
            if trusted_corporate_action_ledger is not None
            else corporate_action_ledger
        )
        policy = (
            trusted_corporate_action_policy
            if trusted_corporate_action_policy is not None
            else corporate_action_policy
        )
        if ledger is None:
            raise MissingEvidenceError(
                "trusted corporate-action evidence ledger is required for integrity validation; "
                "serialized corporate-action evidence is an unverified claim, not proof; "
                "missing trusted ledger => fail closed"
            )
        canonical = tuple(sorted(self.instrument_dataset_fingerprints.keys()))
        try:
            self.corporate_action_evidence.validate_against_trusted_ledger(
                ledger,
                research_start=self.research_window.start,
                research_end=self.research_window.end,
                canonical_instruments=canonical,
                policy=policy,
            )
        except (CorporateActionMismatchError, LookupError, ValueError) as exc:
            raise MissingEvidenceError(
                f"corporate-action evidence integrity mismatch: {exc}"
            ) from exc
        if promotion_thresholds is not None:
            passed, violations = self.evaluate_promotion_gate(
                promotion_thresholds,
                trusted_corporate_action_ledger=ledger,
                trusted_corporate_action_policy=policy,
                **kwargs,
            )
            if not passed:
                raise MissingEvidenceError(
                    "experiment failed promotion gate criteria: " + "; ".join(violations)
                )


# Rebind the core module's runtime globals so its existing orchestrator constructs the hardened
# public artifact while preserving the complete V3 builder implementation and all V3 fields.
_core.CorporateActionEvidenceIdentity = CorporateActionEvidenceIdentity
_core.CorporateActionMismatchError = CorporateActionMismatchError
_core.ExperimentArtifact = ExperimentArtifact
