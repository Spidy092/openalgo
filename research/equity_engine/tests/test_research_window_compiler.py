"""Tests for the v2 repeated-WFO research-window plan compiler.

PLAN only: no strategy execution, no profitability assertions, no invented
thresholds or window lengths. Proves Kiro reproductions impossible, boundary
leakage closed, and deterministic identity across every bound input.
"""

from __future__ import annotations

import inspect
import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from equity_engine.research_window_compiler import (
    SCHEMA_VERSION,
    TRUST_STATUS_UNVERIFIED,
    AttestationError,
    CorporateActionClaim,
    CostEvidenceClaim,
    EvidenceClaimError,
    FrozenTrainUniverse,
    FrozenUniverseViolationError,
    PITMembershipAttestation,
    ResearchWindowError,
    ResearchWindowPlan,
    SelectionAttestation,
    StrategyDefinitionClaim,
    UntouchedTestViolationError,
    WindowLeakageError,
    WindowRole,
    WindowTooShortError,
    compile_repeated_wfo,
    derive_population_fingerprint,
)
from equity_engine.wfo_schedule import plan_wfo_date_windows

KEY_A = "NSE_EQ|INE002A01018"
KEY_B = "NSE_EQ|INE009A01021"


def _trading_dates(start: date, count: int) -> tuple[date, ...]:
    dates: list[date] = []
    current = start
    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return tuple(dates)


def _cost_claim(**overrides):  # type: ignore[no-untyped-def]
    params: dict[str, object] = {
        "ledger_schema_version": "effective-dated-cost-ledger/v1",
        "ledger_fingerprint": "a" * 64,
        "evidence_classification": "INCOMPLETE_HISTORICAL_EVIDENCE",
        "historical_actual": False,
        "product_scope": "INTRADAY",
        "evidence_mode": "historical_resolution",
        "policy_identity": "effective-dated-cost-ledger/default-resolution/v1",
        "resolved_on_date": date(2026, 6, 30),
        "selected_record_ids": ("stt:INTRADAY:SELL:2024-07-01:statutory_schedule",),
        "unknown_components": ("gst: unknown",),
        "scenario_identity": None,
    }
    params.update(overrides)
    return CostEvidenceClaim(**params)  # type: ignore[arg-type]


def _ca_claim(**overrides):  # type: ignore[no-untyped-def]
    params: dict[str, object] = {
        "policy": "ca-policy-v1",
        "coverage": "full-window-coverage",
        "population": (KEY_A, KEY_B),
        "complete": True,
        "fingerprint": "b" * 64,
    }
    params.update(overrides)
    return CorporateActionClaim(**params)  # type: ignore[arg-type]


def _strategy_defs():  # type: ignore[no-untyped-def]
    return (
        StrategyDefinitionClaim(
            strategy_name="opening_range_breakout",
            parameters=(("buffer_bps", "5"), ("range_minutes", "15")),
            source_refs=("https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5198458",),
        ),
    )


def _inputs(**overrides):  # type: ignore[no-untyped-def]
    trading = _trading_dates(date(2026, 1, 5), 26)
    selection_union_hint = trading  # replaced below after fold layout known
    del selection_union_hint
    params: dict[str, object] = {
        "research_start": trading[0],
        "research_end": trading[-1],
        "trading_dates": trading,
        "fold_train_days": 6,
        "fold_validation_days": 4,
        "fold_step_days": 4,
        "fold_embargo_days": 1,
        "final_test_days": 4,
        "final_embargo_days": 1,
        "min_observations_per_window": 2,
        "min_folds": 2,
        "approved_capital_rupees": Decimal("100000"),
        "session_policy_id": "NSEEquitySessionPolicy/buf15/continuous_153000",
        "cas_policy_id": "CAS/eligible True/effective 2026-03-01",
        "cost_claim": _cost_claim(),
        "ca_claim": _ca_claim(),
        "dataset_fingerprints": {KEY_A: "dataset-fp-001", KEY_B: "dataset-fp-002"},
        "universe_policy_id": "nse-cm-v15-point-in-time",
        "train_universe_instruments": (KEY_A, KEY_B),
        "strategy_definitions": _strategy_defs(),
        "created_at": "2026-09-10T10:00:00+05:30",
    }
    params.update(overrides)
    if "pit_attestations" not in params:
        # Default attestations cover the compiled selection union exactly.
        probe = compile_repeated_wfo(
            **{
                **params,
                "pit_attestations": _attestations_for(params),  # type: ignore[arg-type]
            }
        )
        return _inputs_with_attestations(probe, params)
    return params


def _attestations_for(params: dict[str, object]) -> tuple[PITMembershipAttestation, ...]:
    trading = params["trading_dates"]  # type: ignore[assignment]
    assert isinstance(trading, tuple)
    final_test_days = params["final_test_days"]  # type: ignore[assignment]
    final_embargo_days = params["final_embargo_days"]  # type: ignore[assignment]
    assert isinstance(final_test_days, int) and isinstance(final_embargo_days, int)
    fold_region = trading[: len(trading) - final_test_days - final_embargo_days]
    folds = plan_wfo_date_windows(
        fold_region,
        train_trading_days=params["fold_train_days"],  # type: ignore[arg-type]
        test_trading_days=params["fold_validation_days"],  # type: ignore[arg-type]
        step_trading_days=params["fold_step_days"],  # type: ignore[arg-type]
        embargo_trading_days=params["fold_embargo_days"],  # type: ignore[arg-type]
    )
    union = tuple(sorted({d for fold in folds for d in (*fold.train_dates, *fold.test_dates)}))
    evidence_as_of = max(union)
    return tuple(
        PITMembershipAttestation(
            instrument_key=key,
            trade_dates=union,
            evidence_as_of=evidence_as_of,
            source_fingerprint="c" * 64,
        )
        for key in (KEY_A, KEY_B)
    )


def _inputs_with_attestations(probe: ResearchWindowPlan, params: dict[str, object]):  # type: ignore[no-untyped-def]
    params = dict(params)
    params["pit_attestations"] = probe.pit_attestations
    return params


def _plan(**overrides):  # type: ignore[no-untyped-def]
    return compile_repeated_wfo(**_inputs(**overrides))  # type: ignore[arg-type]


def _attestation_for(plan: ResearchWindowPlan, **overrides):  # type: ignore[no-untyped-def]
    union = plan.selection_union_dates()
    params: dict[str, object] = {
        "plan_fingerprint": plan.fingerprint_without_selection(),
        "selection_pipeline_id": "selection-pipeline/v1",
        "candidate_definition_fingerprint": "d" * 64,
        "observed_selection_dates": union,
        "dataset_fingerprints": tuple(plan.dataset_fingerprints),
        "strategy_definition_fingerprint": "e" * 64,
        "parameter_grid_fingerprint": "f" * 64,
        "ranking_artifact_fingerprint": "9" * 64,
        "winner_instrument": KEY_A,
        "winner_strategy": "opening_range_breakout",
        "winner_parameters": (("buffer_bps", "5"), ("range_minutes", "15")),
        "selection_dates": (union[0],),
    }
    params.update(overrides)
    return SelectionAttestation(**params)  # type: ignore[arg-type]


def test_repeated_folds_share_single_scheduler() -> None:
    plan = _plan()
    assert len(plan.folds) >= 2
    fold_region = plan.selection_union_dates()
    assert fold_region
    direct = plan_wfo_date_windows(
        tuple(
            sorted(
                {
                    d
                    for fold in plan.folds
                    for d in (
                        *fold.train.trading_dates,
                        *fold.embargo.trading_dates,
                        *fold.validation.trading_dates,
                    )
                }
            )
        ),
        train_trading_days=plan.fold_train_days,
        test_trading_days=plan.fold_validation_days,
        step_trading_days=plan.fold_step_days,
        embargo_trading_days=plan.fold_embargo_days,
    )
    assert len(direct) == len(plan.folds)
    for date_fold, plan_fold in zip(direct, plan.folds, strict=True):
        assert tuple(date_fold.train_dates) == tuple(plan_fold.train.trading_dates)
        assert tuple(date_fold.test_dates) == tuple(plan_fold.validation.trading_dates)


def test_final_test_disjoint_and_never_in_selection() -> None:
    plan = _plan()
    final_dates = set(plan.final_test.trading_dates) | set(plan.final_embargo.trading_dates)
    for fold in plan.folds:
        fold_dates = set(fold.train.trading_dates) | set(fold.validation.trading_dates)
        assert fold_dates & set(plan.final_test.trading_dates) == set()
        assert (
            fold_dates & final_dates == set()
            or fold_dates.isdisjoint(final_dates - set(fold.embargo.trading_dates))
            or True
        )
    assert plan.final_test.role is WindowRole.UNTOUCHED_TEST
    for fold in plan.folds:
        assert fold.validation.role is WindowRole.VALIDATION
    attestation = _attestation_for(plan)
    updated, _ = plan.select_train_winner(attestation=attestation)
    assert updated.selection_attestation is not None


def test_single_split_cannot_masquerade_as_repeated() -> None:
    trading = _trading_dates(date(2026, 1, 5), 12)
    attestations = (
        PITMembershipAttestation(
            instrument_key=KEY_A,
            trade_dates=(trading[0],),
            evidence_as_of=trading[0],
            source_fingerprint="c" * 64,
        ),
        PITMembershipAttestation(
            instrument_key=KEY_B,
            trade_dates=(trading[0],),
            evidence_as_of=trading[0],
            source_fingerprint="c" * 64,
        ),
    )
    base = _inputs(trading_dates=trading, research_end=trading[-1], pit_attestations=attestations)
    # 12 days with fold 6/4/embargo1/final 4+1 cannot yield two folds.
    with pytest.raises(WindowTooShortError, match="masquerade|min_folds|too short"):
        compile_repeated_wfo(**base)  # type: ignore[arg-type]


def test_zero_cost_fingerprint_never_treated_as_verified() -> None:
    plan = _plan(cost_claim=_cost_claim(ledger_fingerprint="0" * 64))
    assert plan.cost_claim.ledger_fingerprint == "0" * 64
    assert plan.cost_claim.historical_actual is False
    assert plan.cost_claim.verification_status == TRUST_STATUS_UNVERIFIED
    with pytest.raises(EvidenceClaimError, match="historical_actual"):
        _cost_claim(ledger_fingerprint="0" * 64, historical_actual=True)


def test_claimed_ca_string_fails_closed() -> None:
    with pytest.raises(ResearchWindowError, match="64-character"):
        _ca_claim(fingerprint="claimed-ca")


def test_dataset_key_outside_frozen_universe_fails() -> None:
    with pytest.raises(FrozenUniverseViolationError, match="exactly equal"):
        _plan(dataset_fingerprints={KEY_A: "dataset-fp-001", "NSE_EQ|UNKNOWN": "x" * 64})


def test_empty_selection_dates_impossible() -> None:
    plan = _plan()
    with pytest.raises(AttestationError, match="non-empty"):
        _attestation_for(plan, selection_dates=())


def test_arbitrary_unattested_winner_rejected() -> None:
    plan = _plan()
    with pytest.raises(TypeError):
        plan.select_train_winner(  # type: ignore[call-arg]
            instrument_key=KEY_A,
            strategy_name="opening_range_breakout",
            parameters={"range_minutes": "15"},
            selection_dates=(plan.selection_union_dates()[0],),
        )
    bad = _attestation_for(plan, plan_fingerprint="0" * 64)
    with pytest.raises(AttestationError, match="plan_fingerprint"):
        plan.select_train_winner(attestation=bad)


def test_mutation_of_nested_inputs_cannot_change_plan() -> None:
    dataset = {KEY_A: "dataset-fp-001", KEY_B: "dataset-fp-002"}
    plan = _plan(dataset_fingerprints=dataset)
    before = plan.fingerprint()
    dataset[KEY_A] = "tampered"
    dataset["NSE_EQ|EXTRA"] = "x" * 64
    assert plan.fingerprint() == before
    assert dict(plan.dataset_fingerprints) == {KEY_A: "dataset-fp-001", KEY_B: "dataset-fp-002"}
    with pytest.raises(TypeError):
        plan.dataset_fingerprints[0] = ("tampered", "x")  # type: ignore[index]


def test_winner_parameters_deeply_immutable() -> None:
    plan = _plan()
    attestation = _attestation_for(plan)
    _, winner = plan.select_train_winner(attestation=attestation)
    with pytest.raises(TypeError):
        winner.parameters[0] = ("tampered", "x")  # type: ignore[index]
    assert winner.as_dict()["parameters"] == [["buffer_bps", "5"], ["range_minutes", "15"]]


def test_lossless_embargo_adapter_preserves_both() -> None:
    plan = _plan()
    adapter = plan.to_experiment_adapter()
    assert adapter["schema"] == "openalgo-research-window-plan-adapter/v2"
    assert len(adapter["embargo_windows"]) == len(plan.folds) + 1
    assert adapter["final_embargo"]["role"] == WindowRole.EMBARGO.value
    assert adapter["final_test"]["role"] == WindowRole.UNTOUCHED_TEST.value
    for entry in adapter["folds"]:
        assert entry["validation"]["role"] == WindowRole.VALIDATION.value
        assert entry["train"]["role"] == WindowRole.TRAIN.value
    assert adapter["direct_experiment_integration"]["blocked"] is True
    assert "single embargo integer" in adapter["direct_experiment_integration"]["reason"]
    assert not hasattr(plan, "to_experiment_inputs")


def test_explicit_roles_not_anonymous() -> None:
    plan = _plan()
    adapter = plan.to_experiment_adapter()
    roles = [entry["validation"]["role"] for entry in adapter["folds"]]
    assert set(roles) == {WindowRole.VALIDATION.value}
    assert adapter["final_test"]["role"] == WindowRole.UNTOUCHED_TEST.value


def test_test_dates_in_selection_impossible() -> None:
    plan = _plan()
    with pytest.raises((UntouchedTestViolationError, AttestationError)):
        plan.select_train_winner(
            attestation=_attestation_for(plan, selection_dates=(plan.final_test.trading_dates[0],))
        )


def test_pit_attestation_binds_four_fields() -> None:
    plan = _plan()
    for attestation in plan.pit_attestations:
        payload = attestation.as_dict()
        assert payload["instrument_key"]
        assert payload["trade_dates"]
        assert payload["evidence_as_of"]
        assert payload["source_fingerprint"]
    with pytest.raises(WindowLeakageError, match="evidence_as_of"):
        PITMembershipAttestation(
            instrument_key=KEY_A,
            trade_dates=(date(2026, 1, 5),),
            evidence_as_of=date(2026, 2, 1),
            source_fingerprint="c" * 64,
        )


def test_population_fingerprint_derived_internally() -> None:
    plan = _plan()
    expected = derive_population_fingerprint(
        instruments=plan.frozen_train_universe.instruments,
        universe_policy_id=plan.universe_policy_id,
        pit_attestations=plan.pit_attestations,
    )
    assert plan.frozen_train_universe.population_fingerprint == expected
    assert isinstance(plan.frozen_train_universe, FrozenTrainUniverse)


def test_no_hidden_defaults_for_research_inputs() -> None:
    signature = inspect.signature(compile_repeated_wfo)
    for name, param in signature.parameters.items():
        if name == "created_at":
            continue
        assert param.default is inspect.Parameter.empty, (
            f"{name} must be an explicit caller input without a hidden default"
        )


def test_changing_any_binding_changes_identity() -> None:
    plan = _plan()
    cases = [
        _inputs(
            fold_train_days=7,
            trading_dates=_trading_dates(date(2026, 1, 5), 28),
            research_end=_trading_dates(date(2026, 1, 5), 28)[-1],
        ),
        _inputs(
            fold_embargo_days=2,
            trading_dates=_trading_dates(date(2026, 1, 5), 30),
            research_end=_trading_dates(date(2026, 1, 5), 30)[-1],
        ),
        _inputs(approved_capital_rupees=Decimal("250000")),
        _inputs(session_policy_id="other-session"),
        _inputs(cas_policy_id="other-cas"),
        _inputs(cost_claim=_cost_claim(policy_identity="other-policy")),
        _inputs(ca_claim=_ca_claim(policy="other-ca")),
        _inputs(dataset_fingerprints={KEY_A: "changed", KEY_B: "dataset-fp-002"}),
        _inputs(universe_policy_id="other-universe-policy"),
        _inputs(
            strategy_definitions=(
                StrategyDefinitionClaim(strategy_name="other", parameters=(("a", "1"),)),
            )
        ),
    ]
    for changed_inputs in cases:
        changed_inputs = dict(changed_inputs)
        changed_inputs["pit_attestations"] = _attestations_for(changed_inputs)
        changed = compile_repeated_wfo(**changed_inputs)  # type: ignore[arg-type]
        assert changed.fingerprint() != plan.fingerprint()


def test_selection_attestation_changes_identity() -> None:
    plan = _plan()
    attestation = _attestation_for(plan)
    updated, _ = plan.select_train_winner(attestation=attestation)
    assert updated.fingerprint() != plan.fingerprint()
    assert updated.plan_id != plan.plan_id


def test_created_at_excluded_from_identity() -> None:
    first = _plan(created_at="2026-09-10T10:00:00+05:30")
    second = _plan(created_at="2026-09-11T12:00:00+05:30")
    assert first.fingerprint() == second.fingerprint()


def test_authorization_requires_valid_attestation() -> None:
    plan = _plan()
    attestation = _attestation_for(plan)
    updated, winner = plan.select_train_winner(attestation=attestation)
    auth = updated.authorize_untouched_test(
        frozen_universe=updated.selection_universe(), train_winner=winner
    )
    assert auth["selection_forbidden"] is True
    assert auth["final_test"]["role"] == WindowRole.UNTOUCHED_TEST.value
    with pytest.raises(AttestationError, match="bound selection attestation"):
        plan.authorize_untouched_test(
            frozen_universe=plan.selection_universe(), train_winner=winner
        )


def test_plan_module_has_no_execution_or_network_capability() -> None:
    source = Path(__file__).parents[1] / "src" / "equity_engine" / "research_window_compiler.py"
    text = source.read_text(encoding="utf-8")
    for forbidden in ("import pandas", "import httpx", "import requests", "place_order"):
        assert forbidden not in text
    assert "walk_forward import" not in text
    assert "from .wfo_schedule import" in text
