"""Tests for the eligibility decision artifact (Agent 1, Phase 1).

The eligibility decision is a signed, fail-closed artifact that says strategy X
is approved for live, capital N, instrument Y, valid until date Z. It places no
orders and performs no I/O: build and validate are pure functions of their
inputs plus an explicit clock value.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from equity_engine.eligibility_decision import (
    HARD_CAPITAL_CAP_RUPEES,
    AutonomyMode,
    EligibilityDecision,
    EligibilityError,
    LiveOrderAttemptError,
    Track,
    build_eligibility_decision,
    eligibility_from_dict,
    sign_eligibility_decision,
    validate_eligibility_decision,
)


def _window() -> tuple[datetime, datetime, datetime]:
    now = datetime(2026, 9, 15, 6, 0, tzinfo=UTC)
    return now, now - timedelta(days=1), now + timedelta(days=30)


def _decision(**overrides) -> EligibilityDecision:
    _, valid_from, valid_until = _window()
    params: dict = {
        "strategy_id": "orb-intraday-v1",
        "instrument_key": "NSE_EQ|RELIANCE",
        "track": Track.INTRADAY,
        "approved_capital_rupees": Decimal(1000),
        "per_order_notional_cap_rupees": Decimal(1000),
        "daily_loss_limit_rupees": Decimal(200),
        "valid_from": valid_from,
        "valid_until": valid_until,
        "autonomy_mode": AutonomyMode.SUPERVISED_AUTO,
    }
    params.update(overrides)
    return build_eligibility_decision(**params)


def test_hard_cap_is_one_thousand_rupees() -> None:
    assert HARD_CAPITAL_CAP_RUPEES == Decimal(1000)


def test_happy_path_build_sign_validate_round_trip() -> None:
    now, _, _ = _window()
    decision = _decision()

    assert decision.live_orders_called is False
    signed = sign_eligibility_decision(decision)
    assert signed["live_orders_called"] is False
    assert signed["fingerprint"] == decision.deterministic_fingerprint()

    result = validate_eligibility_decision(signed, now=now)
    assert result.valid
    assert result.violations == ()
    assert result.fingerprint == decision.deterministic_fingerprint()


def test_fingerprint_is_deterministic() -> None:
    first = _decision()
    second = _decision()
    assert first.deterministic_fingerprint() == second.deterministic_fingerprint()
    assert len(first.deterministic_fingerprint()) == 64


def test_exactly_at_cap_is_allowed() -> None:
    decision = _decision(approved_capital_rupees=Decimal(1000))
    assert decision.approved_capital_rupees == Decimal(1000)


def test_capital_over_hard_cap_is_refused_at_build() -> None:
    with pytest.raises(EligibilityError, match="hard cap"):
        _decision(approved_capital_rupees=Decimal("1000.01"))


def test_capital_over_hard_cap_fails_closed_at_validate() -> None:
    now, _, _ = _window()
    signed = sign_eligibility_decision(_decision())
    signed["approved_capital_rupees"] = "5000"
    result = validate_eligibility_decision(signed, now=now)
    assert not result.valid
    assert any("hard cap" in item for item in result.violations)


def test_expired_by_one_second_fails_closed() -> None:
    now, valid_from, _ = _window()
    decision = _decision(valid_from=valid_from, valid_until=now - timedelta(seconds=1))
    signed = sign_eligibility_decision(decision)
    result = validate_eligibility_decision(signed, now=now)
    assert not result.valid
    assert any("expired" in item for item in result.violations)


def test_exactly_at_valid_until_is_still_valid() -> None:
    now, valid_from, _ = _window()
    decision = _decision(valid_from=valid_from, valid_until=now)
    signed = sign_eligibility_decision(decision)
    result = validate_eligibility_decision(signed, now=now)
    assert result.valid


def test_not_yet_valid_fails_closed() -> None:
    now, _, valid_until = _window()
    decision = _decision(valid_from=now + timedelta(seconds=1), valid_until=valid_until)
    signed = sign_eligibility_decision(decision)
    result = validate_eligibility_decision(signed, now=now)
    assert not result.valid
    assert any("not yet valid" in item for item in result.violations)


def test_wrong_instrument_fails_closed() -> None:
    now, _, _ = _window()
    signed = sign_eligibility_decision(_decision())
    result = validate_eligibility_decision(signed, now=now, expected_instrument_key="NSE_EQ|INFY")
    assert not result.valid
    assert any("instrument" in item for item in result.violations)


def test_matching_instrument_passes() -> None:
    now, _, _ = _window()
    signed = sign_eligibility_decision(_decision())
    result = validate_eligibility_decision(
        signed, now=now, expected_instrument_key="NSE_EQ|RELIANCE"
    )
    assert result.valid


def test_missing_field_fails_closed() -> None:
    now, _, _ = _window()
    signed = sign_eligibility_decision(_decision())
    del signed["strategy_id"]
    result = validate_eligibility_decision(signed, now=now)
    assert not result.valid
    assert any("strategy_id" in item for item in result.violations)


def test_tampered_field_breaks_fingerprint_and_fails_closed() -> None:
    now, _, _ = _window()
    signed = sign_eligibility_decision(_decision())
    signed["strategy_id"] = "tampered-strategy"
    result = validate_eligibility_decision(signed, now=now)
    assert not result.valid
    assert any("fingerprint" in item for item in result.violations)


def test_lowered_capital_is_denied() -> None:
    now, _, _ = _window()
    signed = sign_eligibility_decision(_decision())
    signed["approved_capital_rupees"] = "100"
    result = validate_eligibility_decision(signed, now=now)
    assert not result.valid


def test_live_orders_true_in_payload_fails_closed() -> None:
    now, _, _ = _window()
    signed = sign_eligibility_decision(_decision())
    signed["live_orders_called"] = True
    result = validate_eligibility_decision(signed, now=now)
    assert not result.valid
    assert any("live order" in item for item in result.violations)


@pytest.mark.parametrize("smuggled", [True, 1, "false", "False", None, 0, ""])
def test_any_live_orders_value_other_than_literal_false_fails_closed(smuggled) -> None:
    now, _, _ = _window()
    signed = sign_eligibility_decision(_decision())
    signed["live_orders_called"] = smuggled
    result = validate_eligibility_decision(signed, now=now)
    assert not result.valid


def test_missing_live_orders_flag_fails_closed() -> None:
    now, _, _ = _window()
    signed = sign_eligibility_decision(_decision())
    del signed["live_orders_called"]
    result = validate_eligibility_decision(signed, now=now)
    assert not result.valid


def test_constructing_with_live_orders_true_raises() -> None:
    with pytest.raises(LiveOrderAttemptError):
        EligibilityDecision(
            strategy_id="orb-intraday-v1",
            instrument_key="NSE_EQ|RELIANCE",
            track=Track.INTRADAY,
            approved_capital_rupees=Decimal(1000),
            per_order_notional_cap_rupees=Decimal(1000),
            daily_loss_limit_rupees=Decimal(200),
            valid_from=datetime(2026, 9, 1, tzinfo=UTC),
            valid_until=datetime(2026, 10, 1, tzinfo=UTC),
            autonomy_mode=AutonomyMode.SUPERVISED_AUTO,
            live_orders_called=True,
        )


def test_eligibility_from_dict_rejects_live_orders_true() -> None:
    signed = sign_eligibility_decision(_decision())
    signed["live_orders_called"] = True
    with pytest.raises(LiveOrderAttemptError):
        eligibility_from_dict(signed)


def test_per_order_cap_above_approved_capital_is_refused() -> None:
    with pytest.raises(EligibilityError, match="per-order"):
        _decision(
            approved_capital_rupees=Decimal(1000),
            per_order_notional_cap_rupees=Decimal("1000.01"),
        )


def test_non_positive_daily_loss_limit_is_refused() -> None:
    with pytest.raises(EligibilityError, match="daily loss"):
        _decision(daily_loss_limit_rupees=Decimal(0))


def test_daily_loss_limit_above_approved_capital_is_refused() -> None:
    with pytest.raises(EligibilityError, match="daily loss"):
        _decision(
            approved_capital_rupees=Decimal(500),
            per_order_notional_cap_rupees=Decimal(500),
            daily_loss_limit_rupees=Decimal("500.01"),
        )


def test_inverted_validity_window_is_refused() -> None:
    now, _, _ = _window()
    with pytest.raises(EligibilityError, match="valid_from"):
        _decision(valid_from=now + timedelta(days=2), valid_until=now + timedelta(days=1))


def test_unknown_track_is_refused() -> None:
    with pytest.raises(EligibilityError):
        _decision(track="overnight")


def test_unknown_autonomy_mode_is_refused() -> None:
    with pytest.raises(EligibilityError):
        _decision(autonomy_mode="unsupervised")


def test_full_auto_is_explicit_and_round_trips() -> None:
    now, _, _ = _window()
    decision = _decision(autonomy_mode=AutonomyMode.FULL_AUTO)
    signed = sign_eligibility_decision(decision)
    assert signed["autonomy_mode"] == "full_auto"
    assert validate_eligibility_decision(signed, now=now).valid
    assert eligibility_from_dict(signed) == decision


def test_swing_track_is_expressible() -> None:
    decision = _decision(track=Track.SWING)
    assert decision.track is Track.SWING
    assert sign_eligibility_decision(decision)["track"] == "swing"


def test_naive_now_fails_closed() -> None:
    signed = sign_eligibility_decision(_decision())
    # Intentionally naive: the rail must refuse a timestamp it cannot place.
    result = validate_eligibility_decision(signed, now=datetime(2026, 9, 15, 6, 0))  # noqa: DTZ001
    assert not result.valid


def test_non_mapping_payload_fails_closed() -> None:
    now, _, _ = _window()
    result = validate_eligibility_decision(["not", "a", "mapping"], now=now)  # type: ignore[arg-type]
    assert not result.valid


def test_module_has_no_order_or_network_imports() -> None:
    import ast
    import pathlib

    source = (
        pathlib.Path(__file__)
        .resolve()
        .parent.parent.joinpath("src/equity_engine/eligibility_decision.py")
        .read_text()
    )
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(
        name.split(".")[0] in {"broker", "services", "httpx", "requests"} for name in imported
    )
    assert "place_order" not in source
