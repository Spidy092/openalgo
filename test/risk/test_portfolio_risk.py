from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from services.risk import (
    PortfolioIntent,
    PortfolioLimits,
    PortfolioPosition,
    PortfolioRiskCode,
    PortfolioSnapshot,
    SymbolActivity,
    evaluate_portfolio_order,
)

NOW = datetime(2026, 9, 14, 4, 0, tzinfo=timezone.utc)


def limits(**overrides):
    values = dict(
        max_gross_exposure=Decimal("100000"),
        max_abs_net_exposure=Decimal("100000"),
        max_open_positions=5,
        max_symbol_exposure=Decimal("50000"),
        max_symbol_concentration_pct=Decimal("75"),
        max_daily_loss=Decimal("5000"),
        cooldown_seconds=60,
        max_market_data_age_seconds=30,
    )
    values.update(overrides)
    return PortfolioLimits(**values)


def snapshot(**overrides):
    values = dict(
        as_of=NOW,
        positions=(
            PortfolioPosition("INFY", 10, Decimal("1000")),
            PortfolioPosition("HDFCBANK", 5, Decimal("1000")),
        ),
        realized_pnl=Decimal("-1000"),
        unrealized_pnl=Decimal("250"),
        market_open=True,
        market_data_timestamp=NOW - timedelta(seconds=5),
        kill_switch_engaged=False,
        symbol_activity=(),
    )
    values.update(overrides)
    return PortfolioSnapshot(**values)


def intent(**overrides):
    values = dict(
        symbol="TCS",
        side="BUY",
        quantity=5,
        reference_price=Decimal("2000"),
        reduce_only=False,
    )
    values.update(overrides)
    return PortfolioIntent(**values)


def test_normal_entry_is_allowed_and_projection_is_deterministic():
    decision = evaluate_portfolio_order(limits(), snapshot(), intent())

    assert decision.allowed is True
    assert decision.primary_code is PortfolioRiskCode.OK
    assert decision.current_gross_exposure == Decimal("15000")
    assert decision.projected_gross_exposure == Decimal("25000")
    assert decision.projected_net_exposure == Decimal("25000")
    assert decision.projected_open_positions == 3


@pytest.mark.parametrize(
    ("limit_overrides", "expected_code"),
    [
        ({"max_gross_exposure": Decimal("24000")}, PortfolioRiskCode.MAX_GROSS_EXPOSURE),
        ({"max_abs_net_exposure": Decimal("24000")}, PortfolioRiskCode.MAX_NET_EXPOSURE),
        ({"max_open_positions": 2}, PortfolioRiskCode.MAX_OPEN_POSITIONS),
        ({"max_symbol_exposure": Decimal("9000")}, PortfolioRiskCode.MAX_SYMBOL_EXPOSURE),
        (
            {"max_symbol_concentration_pct": Decimal("35")},
            PortfolioRiskCode.MAX_SYMBOL_CONCENTRATION,
        ),
    ],
)
def test_projected_portfolio_limits_reject_new_risk(limit_overrides, expected_code):
    decision = evaluate_portfolio_order(limits(**limit_overrides), snapshot(), intent())

    assert decision.allowed is False
    assert expected_code in decision.codes


def test_daily_loss_and_kill_switch_block_new_risk():
    halted = snapshot(
        realized_pnl=Decimal("-5500"),
        unrealized_pnl=Decimal("0"),
        kill_switch_engaged=True,
    )

    decision = evaluate_portfolio_order(limits(), halted, intent())

    assert decision.allowed is False
    assert PortfolioRiskCode.KILL_SWITCH in decision.codes
    assert PortfolioRiskCode.DAILY_LOSS_LIMIT in decision.codes


def test_verified_reduce_only_exit_is_allowed_through_kill_and_daily_loss_halts():
    halted = snapshot(
        positions=(PortfolioPosition("INFY", 10, Decimal("1000")),),
        realized_pnl=Decimal("-5500"),
        unrealized_pnl=Decimal("0"),
        kill_switch_engaged=True,
    )

    decision = evaluate_portfolio_order(
        limits(max_symbol_concentration_pct=Decimal("60")),
        halted,
        intent(
            symbol="INFY",
            side="SELL",
            quantity=5,
            reference_price=Decimal("1000"),
            reduce_only=True,
        ),
    )

    assert decision.allowed is True
    assert decision.risk_reducing is True
    assert decision.projected_gross_exposure == Decimal("5000")


@pytest.mark.parametrize(
    "bad_intent",
    [
        dict(symbol="INFY", side="BUY", quantity=1, reduce_only=True),
        dict(symbol="INFY", side="SELL", quantity=20, reduce_only=True),
        dict(symbol="NEW", side="SELL", quantity=1, reduce_only=True),
    ],
)
def test_reduce_only_cannot_increase_flip_or_create_a_position(bad_intent):
    state = snapshot(positions=(PortfolioPosition("INFY", 10, Decimal("1000")),))
    decision = evaluate_portfolio_order(
        limits(),
        state,
        intent(reference_price=Decimal("1000"), **bad_intent),
    )

    assert decision.allowed is False
    assert decision.primary_code is PortfolioRiskCode.REDUCE_ONLY_VIOLATION


def test_reduce_only_can_decrease_an_already_breached_gross_limit():
    state = snapshot(
        positions=(PortfolioPosition("INFY", 20, Decimal("1000")),),
        realized_pnl=Decimal("0"),
        unrealized_pnl=Decimal("0"),
    )
    decision = evaluate_portfolio_order(
        limits(
            max_gross_exposure=Decimal("10000"),
            max_symbol_exposure=Decimal("10000"),
            max_symbol_concentration_pct=Decimal("50"),
        ),
        state,
        intent(
            symbol="INFY",
            side="SELL",
            quantity=5,
            reference_price=Decimal("1000"),
            reduce_only=True,
        ),
    )

    assert decision.allowed is True
    assert decision.current_gross_exposure == Decimal("20000")
    assert decision.projected_gross_exposure == Decimal("15000")


@pytest.mark.parametrize(
    ("snapshot_overrides", "expected_code"),
    [
        ({"market_open": None}, PortfolioRiskCode.MARKET_STATE_UNKNOWN),
        ({"market_open": False}, PortfolioRiskCode.MARKET_CLOSED),
        ({"market_data_timestamp": None}, PortfolioRiskCode.MARKET_DATA_MISSING),
        (
            {"market_data_timestamp": NOW - timedelta(seconds=31)},
            PortfolioRiskCode.MARKET_DATA_STALE,
        ),
        (
            {"market_data_timestamp": NOW + timedelta(seconds=1)},
            PortfolioRiskCode.MARKET_DATA_FROM_FUTURE,
        ),
    ],
)
def test_unknown_closed_or_bad_market_state_fails_closed(snapshot_overrides, expected_code):
    decision = evaluate_portfolio_order(limits(), snapshot(**snapshot_overrides), intent())

    assert decision.allowed is False
    assert decision.primary_code is expected_code


def test_symbol_cooldown_blocks_only_new_or_increasing_risk():
    state = snapshot(
        positions=(PortfolioPosition("INFY", 10, Decimal("1000")),),
        symbol_activity=(SymbolActivity("INFY", NOW - timedelta(seconds=10)),),
    )
    increase = evaluate_portfolio_order(
        limits(),
        state,
        intent(symbol="INFY", side="BUY", quantity=1, reference_price=Decimal("1000")),
    )
    reduce = evaluate_portfolio_order(
        limits(),
        state,
        intent(
            symbol="INFY",
            side="SELL",
            quantity=1,
            reference_price=Decimal("1000"),
            reduce_only=True,
        ),
    )

    assert increase.allowed is False
    assert PortfolioRiskCode.COOLDOWN in increase.codes
    assert reduce.allowed is True


def test_duplicate_symbol_snapshot_fails_closed():
    state = snapshot(
        positions=(
            PortfolioPosition("INFY", 10, Decimal("1000")),
            PortfolioPosition(" infy ", 1, Decimal("1000")),
        )
    )

    decision = evaluate_portfolio_order(limits(), state, intent())

    assert decision.allowed is False
    assert decision.primary_code is PortfolioRiskCode.INVALID_SNAPSHOT
    assert "duplicate symbol INFY" in decision.reasons[0]


def test_limits_reject_invalid_configuration():
    with pytest.raises(ValueError, match="max_open_positions"):
        limits(max_open_positions=0)
    with pytest.raises(ValueError, match="max_symbol_concentration_pct"):
        limits(max_symbol_concentration_pct=Decimal("101"))
    with pytest.raises(ValueError, match="max_abs_net_exposure"):
        limits(max_abs_net_exposure=Decimal("0"))
