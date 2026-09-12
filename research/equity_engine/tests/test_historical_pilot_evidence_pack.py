from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

_BUILDER_PATH = Path(__file__).parents[1] / "scripts" / "build_historical_pilot_evidence_pack.py"
_BUILDER_SPEC = importlib.util.spec_from_file_location(
    "historical_pilot_evidence_pack_builder", _BUILDER_PATH
)
assert _BUILDER_SPEC is not None and _BUILDER_SPEC.loader is not None
builder = importlib.util.module_from_spec(_BUILDER_SPEC)
_BUILDER_SPEC.loader.exec_module(builder)


def _candidate_identity(manifest: dict[str, object]) -> dict[str, object]:
    candidate = dict(manifest["selected_candidates"][0])  # type: ignore[index]
    candidate.pop("corroborative_current_metadata", None)
    return candidate


def test_current_bod_is_nonblocking_and_outside_historical_identity() -> None:
    base_plan, base_evidence, base_context = builder._build_plan()
    base_manifest = builder._candidate_manifest(base_plan, base_context)

    changed_bod = copy.deepcopy(builder.BOD_SOURCE)
    changed_bod["instrument_record"]["cas_eligible"] = False
    changed_plan, changed_evidence, changed_context = builder._build_plan(
        current_bod=changed_bod,
    )
    changed_manifest = builder._candidate_manifest(changed_plan, changed_context)

    removed_plan, removed_evidence, removed_context = builder._build_plan(current_bod=None)
    removed_manifest = builder._candidate_manifest(removed_plan, removed_context)

    assert changed_plan.plan_id == base_plan.plan_id == removed_plan.plan_id
    assert changed_plan.deterministic_fingerprint() == base_plan.deterministic_fingerprint()
    assert removed_plan.deterministic_fingerprint() == base_plan.deterministic_fingerprint()
    assert changed_evidence.as_dict() == base_evidence.as_dict()
    assert removed_evidence.as_dict() == base_evidence.as_dict()
    assert changed_manifest["manifest_fingerprint"] == base_manifest["manifest_fingerprint"]
    assert removed_manifest["manifest_fingerprint"] == base_manifest["manifest_fingerprint"]
    assert _candidate_identity(changed_manifest) == _candidate_identity(base_manifest)
    assert _candidate_identity(removed_manifest) == _candidate_identity(base_manifest)
    assert changed_context["current_metadata"]["classification"] == (
        "CORROBORATIVE_CURRENT_METADATA"
    )
    assert removed_context["current_metadata"]["present"] is False


def test_dated_nse_cas_evidence_controls_selection() -> None:
    changed_mii = copy.deepcopy(builder.MII_SOURCE)
    changed_mii["2026-08-03"]["raw_fields"]["ElgbltyClsgAuctnSsn"] = "0"

    with pytest.raises(ValueError, match="dated NSE CAS/listing evidence failed"):
        builder._build_plan(mii_source=changed_mii)


def test_special_session_audit_example_is_outside_pilot_window(tmp_path: Path) -> None:
    result = builder.build_pack(tmp_path)
    plan = json.loads((tmp_path / "historical_acquisition_plan.json").read_text())
    special = json.loads((tmp_path / "excluded_special_session.json").read_text())
    summary = (tmp_path / "pilot_summary.md").read_text()

    assert result["live_orders_called"] is False
    assert plan["excluded_special_session_dates"] == []
    assert special["classification"] == "OUTSIDE_PILOT_WINDOW"
    assert special["session_classification"] == "SPECIAL_SESSION_EXCLUDED"
    assert special["counted_as_instrument_trading_day"] is False
    assert "OUTSIDE_PILOT_WINDOW" in summary
    assert "NOT_COUNTED" in summary
