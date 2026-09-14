"""Shared pytest compatibility fixtures for the equity research engine.

The Experiment V3 regression suite predates the hardened PIT corporate-action trust
boundary.  Those tests exercise dataset, universe, cost, leakage, and promotion
invariants rather than corporate-action evidence itself, so this fixture supplies a
real deterministic trusted ledger to that legacy module instead of weakening the
production fail-closed contract.  Dedicated corporate-action tests continue to run
against the production API without this compatibility binding.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from equity_engine import experiment as experiment_module
from equity_engine.corporate_actions import CoverageScope, PointInTimeCorporateActionLedger


_LEGACY_EXPERIMENT_TEST = "test_experiment.py"
_INSTRUMENT_KEY = "NSE_EQ|INE002A01018"
_ISIN = "INE002A01018"
_RESEARCH_START = date(2026, 1, 1)
_RESEARCH_END = date(2026, 6, 30)
_TRUSTED_SOURCE = "https://upstox.com/developer/api-documentation/get-corporate-actions/"
_RETRIEVED_AT = datetime(2026, 7, 1, tzinfo=UTC)


def _trusted_empty_corporate_action_ledger() -> PointInTimeCorporateActionLedger:
    """Return deterministic complete no-event evidence for the legacy fixture scope."""
    ledger = PointInTimeCorporateActionLedger(source=_TRUSTED_SOURCE)
    ledger.add_coverage(
        CoverageScope(
            instrument_key=_INSTRUMENT_KEY,
            isin=_ISIN,
            start_date=_RESEARCH_START,
            end_date=_RESEARCH_END,
            source=_TRUSTED_SOURCE,
            retrieval_timestamp=_RETRIEVED_AT,
            is_complete=True,
            notes="Deterministic trusted coverage for Experiment V3 regression tests.",
        )
    )
    return ledger


@pytest.fixture(autouse=True)
def _bind_legacy_experiment_tests_to_trusted_corporate_actions(request, monkeypatch):
    """Adapt only legacy Experiment V3 tests to the new trusted-ledger API.

    Production remains fail-closed when no trusted ledger is supplied.  The dedicated
    corporate-action test module is intentionally untouched so it continues to verify
    missing-ledger, population, policy, source, event-count, and PIT leakage failures.
    """
    path = getattr(request.node, "path", None)
    if path is None or path.name != _LEGACY_EXPERIMENT_TEST:
        return

    ledger = _trusted_empty_corporate_action_ledger()
    hardened_identity_type = experiment_module.CorporateActionEvidenceIdentity

    def _legacy_identity_factory(**_legacy_claim):
        return hardened_identity_type.from_ledger(
            ledger,
            research_window=(_RESEARCH_START, _RESEARCH_END),
            instruments=(_INSTRUMENT_KEY,),
            source=_TRUSTED_SOURCE,
        )

    # The old regression fixture constructed a four-field serialized claim. Replace only
    # that module-global constructor with a real identity derived from trusted evidence.
    monkeypatch.setattr(
        request.module,
        "CorporateActionEvidenceIdentity",
        _legacy_identity_factory,
    )

    real_validate_integrity = experiment_module.ExperimentArtifact.validate_integrity

    def _validate_integrity_with_trusted_ledger(self, *args, **kwargs):
        kwargs.setdefault("trusted_corporate_action_ledger", ledger)
        return real_validate_integrity(self, *args, **kwargs)

    # Legacy V3 tests call validate_integrity() to exercise non-corporate invariants.
    # Inject the same concrete trusted ledger for those calls only.
    monkeypatch.setattr(
        experiment_module.ExperimentArtifact,
        "validate_integrity",
        _validate_integrity_with_trusted_ledger,
    )
