"""Tests for the deterministic research-window plan compiler.

PLAN only: no strategy execution, no profitability assertions, no invented
thresholds or window lengths. Covers boundary leakage and deterministic
identity, plus fail-closed behavior for invalid or too-short windows.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from equity_engine.research_window_compiler import (
    SCHEMA_VERSION,
    FrozenUniverseViolationError,
    ResearchWindowError,
    ResearchWindowPlan,
    UntouchedTestViolationError,
    WindowLeakageError,
    WindowRole,
    WindowTooShortError,
    compile_research_windows,
)


def _trading_dates(start: date, count: int) -> tuple[date, ...]:
    dates: list[date] = []
    current = start
    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return tuple(dates)


def _inputs(**overrides):  # type: ignore[no-untyped-def]
    start = date(2026, 1, 5)
    # train 10 + embargo 2 + validation 6 + embargo 2 + test 6 = 26 days.
    trading = _trading_dates(start, 26)
    params: dict[str, object] = {
        "research_start": trading[0],
        "research_end": trading[-1],
        "trading_dates": trading,
        "train_trading_days": 10,
        "validation_trading_days": 6,
        "test_trading_days": 6,
        "embargo_trading_days": 2,
        "min_observations_per_window": 3,
        "cost_evidence_fingerprint": "a" * 64,
        "approved_capital_rupees": Decimal("100000"),
        "session_policy_id": "NSEEquitySessionPolicy/buf15/continuous_153000",
        "cas_policy_id": "CAS/eligible True/effective 2026-03-01",
        "corporate_action_evidence_fingerprint": "ca-fingerprint-001",
        "dataset_fingerprints": {"NSE_EQ|INE002A01018": "dataset-fp-001"},
        "universe_fingerprint": "universe-fp-001",
        "pit_membership_fingerprint": "pit-fp-001",
        "train_universe_instruments": ("NSE_EQ|INE002A01018", "NSE_EQ|INE009A01021"),
        "created_at": "2026-09-10T10:00:00+05:30",
    }
    params.update(overrides)
    return params


def _plan(**overrides):  # type: ignore[no-untyped-def]
    return compile_research_windows(**_inputs(**overrides))  # type: ignore[arg-type]


def test_compile_produces_chronological_windows() -> None:
    plan = _plan()

    assert plan.schema_version == SCHEMA_VERSION
    assert plan.train.role is WindowRole.TRAIN
    assert plan.validation.role is WindowRole.VALIDATION
    assert plan.untouched_test.role is WindowRole.UNTOUCHED_TEST
    assert plan.train.window_id == "train_01"
    assert plan.validation.window_id == "validation_01"
    assert plan.untouched_test.window_id == "test_untouched_01"
    assert len(plan.train.trading_dates) == 10
    assert len(plan.validation.trading_dates) == 6
    assert len(plan.untouched_test.trading_dates) == 6
    assert len(plan.embargo_after_train.trading_dates) == 2
    assert len(plan.embargo_after_validation.trading_dates) == 2
    assert plan.train.end < plan.embargo_after_train.start  # type: ignore[operator]
    assert plan.embargo_after_train.end < plan.validation.start  # type: ignore[operator]
    assert plan.validation.end < plan.embargo_after_validation.start  # type: ignore[operator]
    assert plan.embargo_after_validation.end < plan.untouched_test.start  # type: ignore[operator]


def test_no_overlap_through_embargo() -> None:
    plan = _plan()
    seen: set[date] = set()
    for window in (
        plan.train,
        plan.embargo_after_train,
        plan.validation,
        plan.embargo_after_validation,
        plan.untouched_test,
    ):
        overlap = seen & set(window.trading_dates)
        assert overlap == set()
        seen |= set(window.trading_dates)


def test_too_short_calendar_fails_closed() -> None:
    trading = _trading_dates(date(2026, 1, 5), 25)
    with pytest.raises(WindowTooShortError, match="require"):
        compile_research_windows(
            **_inputs(trading_dates=trading, research_end=trading[-1])  # type: ignore[arg-type]
        )


def test_leftover_calendar_fails_closed_without_hidden_truncation() -> None:
    trading = _trading_dates(date(2026, 1, 5), 27)
    with pytest.raises(WindowTooShortError):
        compile_research_windows(
            **_inputs(trading_dates=trading, research_end=trading[-1])  # type: ignore[arg-type]
        )


def test_misaligned_calendar_fails_closed() -> None:
    trading = _trading_dates(date(2026, 1, 5), 26)
    shifted = tuple(d + timedelta(days=1) for d in trading)
    with pytest.raises(ResearchWindowError, match="exactly span"):
        compile_research_windows(**_inputs(trading_dates=shifted))  # type: ignore[arg-type]


def test_unsorted_calendar_fails_closed() -> None:
    inputs = _inputs()
    trading = list(inputs["trading_dates"])  # type: ignore[union-attr]
    trading[0], trading[1] = trading[1], trading[0]
    with pytest.raises(ResearchWindowError, match="sorted unique"):
        compile_research_windows(**_inputs(trading_dates=tuple(trading)))  # type: ignore[arg-type]


def test_min_observations_guard_fails_closed() -> None:
    with pytest.raises(WindowTooShortError, match="min_observations"):
        _plan(test_trading_days=2, min_observations_per_window=3)


def test_invalid_durations_fail_closed() -> None:
    with pytest.raises(ResearchWindowError, match="positive integer"):
        _plan(train_trading_days=0)
    with pytest.raises(ResearchWindowError, match="non-negative integer"):
        _plan(embargo_trading_days=-1)
    with pytest.raises(ResearchWindowError, match="positive integer"):
        _plan(min_observations_per_window=0)


def test_no_hidden_defaults_for_research_inputs() -> None:
    signature = inspect.signature(compile_research_windows)
    for name, param in signature.parameters.items():
        if name == "created_at":
            continue
        assert param.default is inspect.Parameter.empty, (
            f"{name} must be an explicit caller input without a hidden default"
        )


def test_untouched_test_cannot_enter_selection() -> None:
    plan = _plan()
    leaked = (plan.untouched_test.trading_dates[0],)
    with pytest.raises(UntouchedTestViolationError, match="untouched test"):
        plan.select_train_winner(
            instrument_key="NSE_EQ|INE002A01018",
            strategy_name="opening_range_breakout",
            parameters={"range_minutes": "15"},
            selection_dates=leaked,
        )


def test_embargo_dates_cannot_enter_selection() -> None:
    plan = _plan()
    leaked = (plan.embargo_after_train.trading_dates[0],)
    with pytest.raises(UntouchedTestViolationError):
        plan.select_train_winner(
            instrument_key="NSE_EQ|INE002A01018",
            strategy_name="opening_range_breakout",
            parameters={"range_minutes": "15"},
            selection_dates=leaked,
        )


def test_selection_outside_train_validation_fails_closed() -> None:
    plan = _plan()
    with pytest.raises(WindowLeakageError, match="train and validation"):
        plan.select_train_winner(
            instrument_key="NSE_EQ|INE002A01018",
            strategy_name="opening_range_breakout",
            parameters={"range_minutes": "15"},
            selection_dates=(date(2025, 12, 31),),
        )


def test_train_winner_requires_stock_strategy_params() -> None:
    plan = _plan()
    allowed = (plan.train.trading_dates[0],)
    with pytest.raises(ResearchWindowError, match="instrument_key"):
        plan.select_train_winner(
            instrument_key="  ",
            strategy_name="opening_range_breakout",
            parameters={"range_minutes": "15"},
            selection_dates=allowed,
        )
    with pytest.raises(ResearchWindowError, match="strategy_name"):
        plan.select_train_winner(
            instrument_key="NSE_EQ|INE002A01018",
            strategy_name="  ",
            parameters={"range_minutes": "15"},
            selection_dates=allowed,
        )
    with pytest.raises(ResearchWindowError, match="parameters"):
        plan.select_train_winner(
            instrument_key="NSE_EQ|INE002A01018",
            strategy_name="opening_range_breakout",
            parameters={},
            selection_dates=allowed,
        )


def test_selection_rejects_stock_outside_frozen_universe() -> None:
    plan = _plan()
    allowed = (plan.train.trading_dates[0],)
    with pytest.raises(FrozenUniverseViolationError, match="frozen train universe"):
        plan.select_train_winner(
            instrument_key="NSE_EQ|UNKNOWN",
            strategy_name="opening_range_breakout",
            parameters={"range_minutes": "15"},
            selection_dates=allowed,
        )


def test_happy_path_selection_then_untouched_authorization() -> None:
    plan = _plan()
    selection_dates = plan.train.trading_dates[:2] + plan.validation.trading_dates[:1]
    winner = plan.select_train_winner(
        instrument_key="NSE_EQ|INE002A01018",
        strategy_name="opening_range_breakout",
        parameters={"range_minutes": "15", "volume_ratio": "1.5"},
        selection_dates=selection_dates,
    )
    assert winner.instrument_key == "NSE_EQ|INE002A01018"
    assert winner.strategy_name == "opening_range_breakout"
    assert winner.parameters == {"range_minutes": "15", "volume_ratio": "1.5"}

    frozen = plan.selection_universe()
    assert tuple(sorted(frozen.instruments)) == frozen.instruments
    assert frozen.frozen_as_of == plan.train.end

    auth = plan.authorize_untouched_test(frozen_universe=frozen, train_winner=winner)
    assert auth["plan_id"] == plan.plan_id
    assert auth["selection_forbidden"] is True
    assert auth["frozen_train_universe"]["instruments"] == list(frozen.instruments)
    assert auth["train_winner"] == winner.as_dict()
    assert auth["test_window"]["window_id"] == "test_untouched_01"


def test_untouched_test_requires_full_frozen_universe() -> None:
    plan = _plan()
    winner = plan.select_train_winner(
        instrument_key="NSE_EQ|INE002A01018",
        strategy_name="opening_range_breakout",
        parameters={"range_minutes": "15"},
        selection_dates=(plan.train.trading_dates[0],),
    )
    tampered = replace(plan.frozen_train_universe, instruments=("NSE_EQ|INE002A01018",))
    with pytest.raises(FrozenUniverseViolationError, match="full frozen"):
        plan.authorize_untouched_test(frozen_universe=tampered, train_winner=winner)


def test_pit_membership_is_per_date_and_rejects_future_evidence() -> None:
    plan = _plan()
    trade_date = plan.validation.trading_dates[0]
    plan.check_pit_membership(trade_date=trade_date, evidence_as_of=trade_date)
    with pytest.raises(WindowLeakageError, match="PIT membership"):
        plan.check_pit_membership(
            trade_date=trade_date, evidence_as_of=plan.untouched_test.trading_dates[0]
        )
    with pytest.raises(ResearchWindowError, match="outside the compiled plan"):
        plan.check_pit_membership(trade_date=date(2025, 1, 1), evidence_as_of=date(2025, 1, 1))


def test_deterministic_identity() -> None:
    first = _plan()
    second = _plan()
    assert first.fingerprint() == second.fingerprint()
    assert first.plan_id == second.plan_id
    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == json.loads(second.to_json())


def test_created_at_does_not_change_identity() -> None:
    first = _plan(created_at="2026-09-10T10:00:00+05:30")
    second = _plan(created_at="2026-09-11T12:00:00+05:30")
    assert first.fingerprint() == second.fingerprint()
    assert first.plan_id == second.plan_id
    assert first.to_dict()["created_at"] != second.to_dict()["created_at"]


@pytest.mark.parametrize(
    "field",
    [
        "cost_evidence_fingerprint",
        "approved_capital_rupees",
        "session_policy_id",
        "cas_policy_id",
        "corporate_action_evidence_fingerprint",
        "dataset_fingerprints",
        "universe_fingerprint",
        "pit_membership_fingerprint",
        "train_universe_instruments",
        "train_trading_days",
        "validation_trading_days",
        "test_trading_days",
        "embargo_trading_days",
        "min_observations_per_window",
    ],
)
def test_changing_any_binding_changes_identity(field: str) -> None:
    plan = _plan()
    if field == "cost_evidence_fingerprint":
        changed = _plan(cost_evidence_fingerprint="b" * 64)
    elif field == "approved_capital_rupees":
        changed = _plan(approved_capital_rupees=Decimal("250000"))
    elif field == "session_policy_id":
        changed = _plan(session_policy_id="NSEEquitySessionPolicy/buf10/continuous_153000")
    elif field == "cas_policy_id":
        changed = _plan(cas_policy_id="CAS/eligible False/effective 2026-03-01")
    elif field == "corporate_action_evidence_fingerprint":
        changed = _plan(corporate_action_evidence_fingerprint="ca-fingerprint-002")
    elif field == "dataset_fingerprints":
        changed = _plan(dataset_fingerprints={"NSE_EQ|INE002A01018": "dataset-fp-002"})
    elif field == "universe_fingerprint":
        changed = _plan(universe_fingerprint="universe-fp-002")
    elif field == "pit_membership_fingerprint":
        changed = _plan(pit_membership_fingerprint="pit-fp-002")
    elif field == "train_universe_instruments":
        changed = _plan(train_universe_instruments=("NSE_EQ|INE002A01018", "NSE_EQ|INE030A01027"))
    elif field == "train_trading_days":
        trading = _trading_dates(date(2026, 1, 5), 27)
        changed = _plan(
            trading_dates=trading,
            research_end=trading[-1],
            train_trading_days=11,
        )
    elif field == "validation_trading_days":
        trading = _trading_dates(date(2026, 1, 5), 27)
        changed = _plan(
            trading_dates=trading,
            research_end=trading[-1],
            validation_trading_days=7,
        )
    elif field == "test_trading_days":
        trading = _trading_dates(date(2026, 1, 5), 27)
        changed = _plan(
            trading_dates=trading,
            research_end=trading[-1],
            test_trading_days=7,
        )
    elif field == "embargo_trading_days":
        trading = _trading_dates(date(2026, 1, 5), 28)
        changed = _plan(
            trading_dates=trading,
            research_end=trading[-1],
            embargo_trading_days=3,
        )
    elif field == "min_observations_per_window":
        changed = _plan(min_observations_per_window=4)
    else:  # pragma: no cover
        raise AssertionError(field)
    assert changed.fingerprint() != plan.fingerprint()
    assert changed.plan_id != plan.plan_id


def test_window_boundary_shift_changes_identity() -> None:
    plan = _plan()
    shifted_start = date(2026, 1, 6)
    # Same durations but shifted overall range: still 26 weekdays from Jan 6.
    trading = _trading_dates(shifted_start, 26)
    shifted = _plan(research_start=trading[0], research_end=trading[-1], trading_dates=trading)
    assert shifted.fingerprint() != plan.fingerprint()


def test_experiment_integration_mapping() -> None:
    plan = _plan()
    inputs = plan.to_experiment_inputs()
    assert inputs["research_window"] == {
        "start": plan.research_start.isoformat(),
        "end": plan.research_end.isoformat(),
    }
    assert inputs["train_windows"][0]["trading_days"] == 10
    assert [w["trading_days"] for w in inputs["validation_test_windows"]] == [6, 6]
    assert inputs["embargo"] == {"trading_days": 2}
    # Train window ends strictly before validation starts; validation before test.
    assert inputs["train_windows"][0]["end"] < inputs["validation_test_windows"][0]["start"]
    assert (
        inputs["validation_test_windows"][0]["end"] < inputs["validation_test_windows"][1]["start"]
    )


def test_plan_module_has_no_execution_or_network_capability() -> None:
    source = Path(__file__).parents[1] / "src" / "equity_engine" / "research_window_compiler.py"
    text = source.read_text(encoding="utf-8")
    for forbidden in ("import pandas", "import httpx", "import requests", "place_order"):
        assert forbidden not in text
    assert "random" not in text.lower() or "no random" in text.lower()
    assert "profit" not in text.lower() or "No strategy execution" in text


def test_plan_is_json_serializable_and_deterministic() -> None:
    plan = _plan()
    assert plan.to_dict() == json.loads(json.dumps(plan.to_dict()))
    assert isinstance(plan, ResearchWindowPlan)
