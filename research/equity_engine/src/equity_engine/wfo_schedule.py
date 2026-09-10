"""Pure chronological walk-forward date scheduler (pandas-free).

Single source of truth for repeated walk-forward fold arithmetic. Both
``walk_forward.make_walk_forward_windows`` (frame-based, pandas) and
``research_window_compiler`` (plan-only, no data) delegate here so no second
scheduler exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class DateFold:
    """One chronological fold: train dates, embargo gap, test dates."""

    window_id: int
    train_dates: tuple[date, ...]
    test_dates: tuple[date, ...]


def plan_wfo_date_windows(
    trading_dates: tuple[date, ...],
    *,
    train_trading_days: int,
    test_trading_days: int,
    step_trading_days: int,
    embargo_trading_days: int,
) -> tuple[DateFold, ...]:
    """Return deterministic repeated folds over sorted unique trading dates.

    Exact stepping algorithm shared with ``walk_forward``: start at 0,
    ``train_end = start + train``, ``test_start = train_end + embargo``,
    ``test_end = test_start + test``; stop when ``test_end`` exceeds the
    calendar; advance ``start`` by ``step``. Fails closed when no fold fits.
    """
    for name, value in (
        ("train_trading_days", train_trading_days),
        ("test_trading_days", test_trading_days),
        ("step_trading_days", step_trading_days),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        not isinstance(embargo_trading_days, int)
        or isinstance(embargo_trading_days, bool)
        or embargo_trading_days < 0
    ):
        raise ValueError("embargo_trading_days must be a non-negative integer")
    ordered = tuple(trading_dates)
    if tuple(sorted(set(ordered))) != ordered:
        raise ValueError("trading_dates must be sorted unique dates")
    if not ordered:
        raise ValueError("trading_dates must not be empty")

    folds: list[DateFold] = []
    start = 0
    window_id = 1
    while True:
        train_end = start + train_trading_days
        test_start = train_end + embargo_trading_days
        test_end = test_start + test_trading_days
        if test_end > len(ordered):
            break
        folds.append(
            DateFold(
                window_id=window_id,
                train_dates=ordered[start:train_end],
                test_dates=ordered[test_start:test_end],
            )
        )
        start += step_trading_days
        window_id += 1
    if not folds:
        raise ValueError("trading dates are too short for requested walk-forward schedule")
    return tuple(folds)
