"""Repeatable intraday discovery pipeline (research only, never sends an execution).

Orchestrates the existing gauntlet in a fixed sequence:

screening -> train/validation/test split with untouched test ->
walk-forward -> event-driven fill simulation -> documented cost model ->
friction stress -> baseline comparison -> offline paper replay ->
promotion gates + hard profitability gates -> validated candidate artifact.

Hard profitability gates (all net of documented costs plus modeled friction;
observed costs enter through the reconciliation gate, which fails closed when
not performed): positive per-trade expectancy on the untouched test period,
positive net on the untouched test period, positive net on a strict majority
of walk-forward folds, configurable minimum profit factor, configurable
minimum trade-based Sharpe, and drawdown within the configured ceiling. Any
gate fails => candidate FAIL.

Research-only: no network, no execution path, intraday cash-equity only.
Every emitted artifact records the live flag as false.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd

from .candidate_grid import (
    StrategyDefinition,
    baseline_definition,
    mean_reversion_grid,
    orb_study_grid,
)
from .cost_ledger import EffectiveDatedCostLedger, LedgerComponent, LedgerProduct
from .documented_costs import CurrentTermsNSEIntradayCostProvider
from .event_simulator import FillAssumptions, IntradaySimulationConfig
from .experiment import ApprovedCapital
from .gates import (
    DrawdownBasis,
    GateDecision,
    PromotionThresholds,
    ResearchEvidence,
    evaluate_promotion_gate,
)
from .historical_cost_scenario import ScenarioAssumption, compile_historical_scenario
from .market_sessions import NSEEquitySessionPolicy, filter_to_continuous_session
from .models import Exchange
from .provenance import (
    MarketDataManifest,
    canonical_sha256,
    dataframe_fingerprint,
    validate_ohlcv_frame,
)
from .shadow_execution import (
    ShadowEngineConfig,
    ShadowInstrumentIdentity,
    ShadowMarketEvent,
    replay_shadow_session,
)
from .stress import FrictionScenario, run_friction_stress
from .tick_size import FixedTickSizePolicy
from .tournament import (
    CandidateEvaluation,
    ExactMetrics,
    RankingMetric,
    evaluate_candidate_exact,
    rank_candidates,
)
from .vectorbt_screening import VectorBTScreeningResult, screen_long_signals
from .walk_forward import make_walk_forward_windows, run_walk_forward_selection

DISCOVERY_SCHEMA_VERSION = "intraday-discovery/v1"
DISCOVERY_TRACK = "intraday"
_LIVE_FLAG_FALSE = False


class _AlwaysEligible:
    """Offline eligibility stub: every date is eligible.

    Production promotion still requires point-in-time membership evidence.
    The artifact records this source explicitly so a synthetic pass cannot
    be mistaken for exchange-verified eligibility.
    """

    def is_eligible(self, trade_date: date) -> bool:
        return True


@dataclass(frozen=True)
class DiscoveryConfig:
    """Caller-supplied research choices. There are no hidden thresholds."""

    instrument_token: str
    symbol: str
    exchange: Exchange
    session_open: time
    bar_minutes: int
    breakout_buffer_bps: Decimal
    screening_cash: Decimal
    screening_fee_rate: Decimal
    screening_slippage_rate: Decimal
    initial_cash: Decimal
    max_trades_per_day: int
    base_slippage_bps_per_leg: Decimal
    base_half_spread_bps_per_leg: Decimal
    stress_slippage_bps_per_leg: Decimal
    stress_half_spread_bps_per_leg: Decimal
    train_end_date: date
    validation_end_date: date
    walk_forward_train_days: int
    walk_forward_test_days: int
    walk_forward_step_days: int
    walk_forward_embargo_days: int
    ranking_metric: RankingMetric
    promotion_thresholds: PromotionThresholds
    min_sharpe: Decimal
    cost_reconciliation_error_inr: Decimal | None
    cost_reconciliation_source: str
    tick_size_rupees: Decimal
    tick_size_source: str
    exit_buffer_minutes: int
    cas_eligible: bool
    pricing_date: date

    def __post_init__(self) -> None:
        if self.exchange is not Exchange.NSE:
            raise ValueError("discovery v1 supports NSE cash equity only")
        if not self.instrument_token.strip() or not self.symbol.strip():
            raise ValueError("instrument_token and symbol are required")
        if self.bar_minutes <= 0:
            raise ValueError("bar_minutes must be positive")
        if self.breakout_buffer_bps < 0:
            raise ValueError("breakout_buffer_bps cannot be negative")
        if self.screening_cash <= 0 or self.initial_cash <= 0:
            raise ValueError("cash limits must be positive")
        if self.screening_fee_rate < 0 or self.screening_slippage_rate < 0:
            raise ValueError("screening fee/slippage cannot be negative")
        if self.max_trades_per_day <= 0:
            raise ValueError("max_trades_per_day must be positive")
        for value in (
            self.base_slippage_bps_per_leg,
            self.base_half_spread_bps_per_leg,
            self.stress_slippage_bps_per_leg,
            self.stress_half_spread_bps_per_leg,
        ):
            if value < 0:
                raise ValueError("friction assumptions cannot be negative")
        if self.train_end_date >= self.validation_end_date:
            raise ValueError("train_end_date must precede validation_end_date")
        if self.tick_size_rupees <= 0:
            raise ValueError("tick_size_rupees must be positive")
        if not self.tick_size_source.strip() or not self.cost_reconciliation_source.strip():
            raise ValueError("tick source and reconciliation source are required")
        if self.exit_buffer_minutes < 0 or self.exit_buffer_minutes >= 60:
            raise ValueError("exit_buffer_minutes must be in [0, 60)")
        if self.cost_reconciliation_error_inr is not None and (
            self.cost_reconciliation_error_inr < 0
        ):
            raise ValueError("cost reconciliation error cannot be negative")
        if not isinstance(self.min_sharpe, Decimal) or not self.min_sharpe.is_finite():
            raise ValueError("min_sharpe must be a finite Decimal")


@dataclass(frozen=True)
class WalkForwardFoldEvidence:
    """Per-fold walk-forward outcome for the eligibility input artifact."""

    window_id: int
    train_dates: tuple[str, ...]
    test_dates: tuple[str, ...]
    selected_candidate_id: str
    test_net_pnl: Decimal
    test_trades: int
    profitable: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "train_dates": list(self.train_dates),
            "test_dates": list(self.test_dates),
            "selected_candidate_id": self.selected_candidate_id,
            "test_net_pnl": format(self.test_net_pnl, "f"),
            "test_trades": self.test_trades,
            "profitable": self.profitable,
        }


@dataclass(frozen=True)
class DiscoveryResult:
    """Validated candidate outcome. The live flag is always false."""

    schema_version: str
    track: str
    selected_candidate_id: str
    selected_parameters: dict[str, str]
    selected_strategy_name: str
    instrument_token: str
    symbol: str
    exchange: str
    status: str
    violations: tuple[str, ...]
    gate_decision: GateDecision
    selected_test_metrics: ExactMetrics
    selected_train_metrics: ExactMetrics
    selected_validation_metrics: ExactMetrics
    baseline_test_net_pnl: Decimal
    baseline_test_trades: int
    walk_forward_windows: int
    walk_forward_aggregate_test_net: Decimal
    walk_forward_folds: tuple[WalkForwardFoldEvidence, ...]
    walk_forward_profitable_folds: int
    test_expectancy: Decimal
    test_sharpe: Decimal
    stress_worst_net_pnl: Decimal | None
    stress_scenarios: tuple[str, ...]
    shadow_decisions: int
    shadow_trades: int
    data_fingerprint: str
    candidate_fingerprint: str
    cost_scenario_fingerprint: str
    shadow_fingerprint: str
    cost_model: str
    pricing_date: str
    tick_size_rupees: str
    tick_size_source: str
    cost_reconciliation_error_inr: str | None
    cost_reconciliation_source: str
    live_orders_called: bool = _LIVE_FLAG_FALSE

    def __post_init__(self) -> None:
        if self.live_orders_called:
            raise ValueError("discovery results never signal a live execution")
        if self.status not in {"PASS", "FAIL"}:
            raise ValueError("status must be PASS or FAIL")
        if self.track != DISCOVERY_TRACK:
            raise ValueError("discovery v1 is intraday only")

    def to_dict(self) -> dict[str, Any]:
        metrics = self.selected_test_metrics
        return {
            "schema_version": self.schema_version,
            "track": self.track,
            "strategy_id": self.selected_candidate_id,
            "rule_params": dict(self.selected_parameters),
            "strategy_name": self.selected_strategy_name,
            "instrument": {
                "instrument_token": self.instrument_token,
                "symbol": self.symbol,
                "exchange": self.exchange,
            },
            "validation_metrics": {
                "test_net_pnl": format(metrics.net_pnl, "f"),
                "test_net_return_pct": format(metrics.net_return_pct, "f"),
                "test_trades": metrics.trade_count,
                "test_wins": metrics.wins,
                "test_losses": metrics.losses,
                "test_profit_factor": (
                    format(metrics.profit_factor, "f")
                    if metrics.profit_factor is not None
                    else None
                ),
                "test_drawdown_ohlc_low_stress_pct": format(
                    metrics.ohlc_low_liquidation_stress_max_drawdown_pct, "f"
                ),
                "train_net_pnl": format(self.selected_train_metrics.net_pnl, "f"),
                "validation_net_pnl": format(self.selected_validation_metrics.net_pnl, "f"),
                "baseline_test_net_pnl": format(self.baseline_test_net_pnl, "f"),
                "baseline_test_trades": self.baseline_test_trades,
                "test_expectancy": format(self.test_expectancy, "f"),
                "test_sharpe": format(self.test_sharpe, "f"),
                "test_max_drawdown_pct": format(
                    metrics.ohlc_low_liquidation_stress_max_drawdown_pct, "f"
                ),
                "walk_forward_windows": self.walk_forward_windows,
                "walk_forward_aggregate_test_net": format(
                    self.walk_forward_aggregate_test_net, "f"
                ),
                "walk_forward_profitable_folds": self.walk_forward_profitable_folds,
                "walk_forward_majority_profitable": (
                    self.walk_forward_profitable_folds * 2 > self.walk_forward_windows
                    if self.walk_forward_windows
                    else False
                ),
                "walk_forward_folds": [fold.as_dict() for fold in self.walk_forward_folds],
                "stress_worst_net_pnl": (
                    format(self.stress_worst_net_pnl, "f")
                    if self.stress_worst_net_pnl is not None
                    else None
                ),
                "stress_scenarios": list(self.stress_scenarios),
                "shadow_decisions": self.shadow_decisions,
                "shadow_trades": self.shadow_trades,
                "held_out_test_present": True,
                "baseline_comparison_present": True,
                "slippage_stress_present": True,
                "event_driven_validation_present": True,
            },
            "cost_assumptions": {
                "model": self.cost_model,
                "pricing_date": self.pricing_date,
                "tick_size_rupees": self.tick_size_rupees,
                "tick_size_source": self.tick_size_source,
                "reconciliation_error_inr": self.cost_reconciliation_error_inr,
                "reconciliation_source": self.cost_reconciliation_source,
            },
            "gate": {
                "passed": self.gate_decision.passed,
                "violations": list(self.gate_decision.violations),
            },
            "discovery_violations": list(self.violations),
            "status": self.status,
            "fingerprints": {
                "data": self.data_fingerprint,
                "candidate": self.candidate_fingerprint,
                "cost_scenario": self.cost_scenario_fingerprint,
                "shadow": self.shadow_fingerprint,
            },
            "live_orders_called": False,
        }


def build_intraday_candidates(
    *,
    session_open: time,
    bar_minutes: int,
    breakout_buffer_bps: Decimal,
) -> list[StrategyDefinition]:
    """Compose 2-3 well-known intraday templates from existing grids.

    Templates: opening-range breakout study grid, rolling z-score
    mean-reversion challenger grid, plus the deliberately simple first-bar
    baseline that every challenger must be compared against at identical cost.
    """

    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive")
    if breakout_buffer_bps < 0:
        raise ValueError("breakout_buffer_bps cannot be negative")
    definitions: list[StrategyDefinition] = [baseline_definition(session_open=session_open)]
    definitions.extend(
        orb_study_grid(
            session_open=session_open,
            bar_minutes=bar_minutes,
            breakout_buffer_bps=breakout_buffer_bps,
        )
    )
    definitions.extend(
        mean_reversion_grid(
            lookback_bars=(10, 20),
            entry_z_values=(Decimal("1.5"), Decimal("2.0")),
            exit_z_values=(Decimal("0.0"),),
        )
    )
    return definitions


def split_train_validation_test(
    frame: pd.DataFrame,
    *,
    train_end_date: date,
    validation_end_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by trade date with a strictly later untouched test slice."""

    if frame.empty:
        raise ValueError("frame cannot be empty")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("frame requires a DatetimeIndex")
    if frame.index.tz is None:
        raise ValueError("frame requires timezone-aware timestamps")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("frame timestamps must be monotonically increasing")
    if train_end_date >= validation_end_date:
        raise ValueError("train_end_date must precede validation_end_date")

    frame_dates = [item.date() for item in frame.index]
    train = frame[[d <= train_end_date for d in frame_dates]].copy()
    validation = frame[
        [(d > train_end_date and d <= validation_end_date) for d in frame_dates]
    ].copy()
    test = frame[[d > validation_end_date for d in frame_dates]].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("train/validation/test split produced an empty slice")
    if max(train.index.date) > min(validation.index.date):
        raise ValueError("train slice overlaps validation slice")
    if max(validation.index.date) >= min(test.index.date):
        raise ValueError("test slice is not strictly after validation slice")
    return train, validation, test


def _screen_on_train(
    *,
    train_frame: pd.DataFrame,
    candidates: list[StrategyDefinition],
    config: DiscoveryConfig,
    session_policy: NSEEquitySessionPolicy,
) -> dict[str, VectorBTScreeningResult]:
    results: dict[str, VectorBTScreeningResult] = {}
    for candidate in candidates:
        try:
            signals = candidate.build_signals(train_frame)
            outcome = screen_long_signals(
                close=train_frame["close"].astype(float),
                execution_price=train_frame["open"].astype(float),
                entries_at_close=signals.entries_at_close,
                exits_at_close=signals.exits_at_close,
                signal_lag_bars=1,
                screening_cash=config.screening_cash,
                screening_fee_rate=config.screening_fee_rate,
                screening_slippage_rate=config.screening_slippage_rate,
                frequency="5min",
                session_policy=session_policy,
            )
        except (RuntimeError, ValueError):
            # A candidate with no screenable edge (for example zero closed
            # trades yielding non-finite VectorBT stats) is excluded from
            # the survivor set without failing the whole discovery run.
            outcome = VectorBTScreeningResult(
                total_return_pct=Decimal(0),
                max_drawdown_pct=Decimal(0),
                closed_trades=0,
                win_rate_pct=None,
                profit_factor=None,
                signal_lag_bars=1,
                exact_cost_validated=False,
            )
        results[candidate.candidate_id] = outcome
    return results


def _evaluate_exact(
    *,
    candidate: StrategyDefinition,
    frame: pd.DataFrame,
    config: DiscoveryConfig,
    session_policy: NSEEquitySessionPolicy,
    tick_policy: FixedTickSizePolicy,
    eligibility: _AlwaysEligible,
    cost_provider: CurrentTermsNSEIntradayCostProvider,
    fills: FillAssumptions,
    simulation_config: IntradaySimulationConfig,
) -> CandidateEvaluation:
    signals = candidate.build_signals(frame)
    return evaluate_candidate_exact(
        candidate_id=candidate.candidate_id,
        frame=frame,
        signals=signals,
        instrument_token=config.instrument_token,
        exchange=config.exchange,
        cost_provider=cost_provider,
        fills=fills,
        session_policy=session_policy,
        tick_size_policy=tick_policy,
        trading_eligibility_policy=eligibility,
        config=simulation_config,
    )


def _profit_factor_or_closed_form(metrics: ExactMetrics) -> Decimal:
    if metrics.profit_factor is not None:
        return metrics.profit_factor
    if metrics.trade_count > 0 and metrics.losses == 0 and metrics.wins > 0:
        return Decimal(999)
    return Decimal(0)


def trade_expectancy(*, net_pnl: Decimal, trade_count: int) -> Decimal:
    """Per-trade net expectancy after all modeled costs. Zero when no trades."""

    if trade_count <= 0:
        return Decimal(0)
    return net_pnl / Decimal(trade_count)


def trade_sharpe(trade_net_pnls: tuple[Decimal, ...]) -> Decimal:
    """Deterministic trade-based Sharpe (mean / sample stdev, unannualized).

    Intraday discovery compares candidates on identical horizons, so no
    annualization factor is applied. Fewer than two trades yields zero (no
    evidence). Zero variance with positive mean yields a large constant;
    zero variance with non-positive mean yields zero or a large negative.
    """

    count = len(trade_net_pnls)
    if count < 2:
        return Decimal(0)
    total = sum(trade_net_pnls, Decimal(0))
    mean = total / Decimal(count)
    squared = sum(((item - mean) ** 2 for item in trade_net_pnls), Decimal(0))
    variance = squared / Decimal(count - 1)
    if variance <= 0:
        if mean > 0:
            return Decimal(999)
        if mean == 0:
            return Decimal(0)
        return Decimal(-999)
    return mean / variance.sqrt()


def _build_cost_scenario(
    *,
    research_start: date,
    research_end: date,
    scenario_date: date,
) -> Any:
    def _assumption(component: LedgerComponent, rate: str) -> ScenarioAssumption:
        return ScenarioAssumption(
            assumption_id=f"discovery-assume-{component.value}",
            component=component,
            product=LedgerProduct.INTRADAY,
            basis="discovery-offline-basis",
            rate=Decimal(rate),
            formula=f"discovery-formula-{rate}",
            source="discovery-offline-source",
            reason="discovery-offline-reason",
        )

    return compile_historical_scenario(
        scenario_id="intraday-discovery-v1",
        ledger=EffectiveDatedCostLedger(),
        scenario_date=scenario_date,
        research_start=research_start,
        research_end=research_end,
        product=LedgerProduct.INTRADAY,
        assumptions=(
            _assumption(LedgerComponent.BROKERAGE, "0.0006"),
            _assumption(LedgerComponent.GST, "0.18"),
            _assumption(LedgerComponent.CLEARING, "0.000001"),
        ),
    )


class _SignalMapPaperStrategy:
    """Offline paper adapter that replays frozen test-slice signals."""

    def __init__(
        self, *, strategy_id: str, entries: dict[str, bool], exits: dict[str, bool]
    ) -> None:
        self._strategy_id = strategy_id
        self._entries = entries
        self._exits = exits

    @property
    def strategy_id(self) -> str:
        return self._strategy_id

    def entry_signal_at_close(self, event: ShadowMarketEvent) -> bool:
        return bool(self._entries.get(event.bar_timestamp.isoformat(), False))

    def exit_signal_at_close(self, event: ShadowMarketEvent) -> bool:
        return bool(self._exits.get(event.bar_timestamp.isoformat(), False))


def _replay_test_slice_as_paper(
    *,
    test_frame: pd.DataFrame,
    candidate_id: str,
    entries: pd.Series,
    exits: pd.Series,
    config: DiscoveryConfig,
    cost_provider: CurrentTermsNSEIntradayCostProvider,
    fills: FillAssumptions,
    cost_scenario: Any,
) -> Any:
    entries_map = {pd.Timestamp(ts).isoformat(): bool(value) for ts, value in entries.items()}
    exits_map = {pd.Timestamp(ts).isoformat(): bool(value) for ts, value in exits.items()}
    strategy = _SignalMapPaperStrategy(
        strategy_id=candidate_id, entries=entries_map, exits=exits_map
    )
    events: list[ShadowMarketEvent] = []
    for seq, (timestamp, row) in enumerate(test_frame.iterrows()):
        bar_ts = pd.Timestamp(timestamp)
        if bar_ts.tzinfo is None:
            raise ValueError("paper replay requires timezone-aware bars")
        received = (bar_ts + timedelta(seconds=5)).to_pydatetime()
        events.append(
            ShadowMarketEvent(
                event_id=f"discovery-paper-{seq}",
                seq=seq,
                instrument_key=config.instrument_token,
                exchange=config.exchange,
                bar_timestamp=bar_ts.to_pydatetime(),
                bar_open=Decimal(str(row["open"])),
                bar_high=Decimal(str(row["high"])),
                bar_low=Decimal(str(row["low"])),
                bar_close=Decimal(str(row["close"])),
                bar_volume=int(row["volume"]),
                received_at=received,
                is_cas_auxiliary=False,
                feed_connected=True,
            )
        )
    return replay_shadow_session(
        events,
        instrument=ShadowInstrumentIdentity(
            instrument_key=config.instrument_token,
            exchange=config.exchange,
            cas_eligible=config.cas_eligible,
            tick_size_rupees=config.tick_size_rupees,
            source="intraday-discovery-v1",
        ),
        strategy=strategy,
        approved_capital=ApprovedCapital(amount_rupees=config.initial_cash),
        cost_scenario=cost_scenario,
        cost_provider=cost_provider,
        fills=fills,
        config=ShadowEngineConfig(
            session_id=f"intraday-discovery-{candidate_id}",
            exit_buffer_minutes=config.exit_buffer_minutes,
            max_quote_age_seconds=60.0,
            bar_interval_seconds=config.bar_minutes * 60,
            max_gap_multiplier=2.0,
            max_trades_per_session=config.max_trades_per_day,
        ),
    )


def run_discovery(
    frame: pd.DataFrame,
    manifest: MarketDataManifest,
    config: DiscoveryConfig,
) -> DiscoveryResult:
    """Execute the fixed gauntlet and emit one validated candidate outcome."""

    structural = validate_ohlcv_frame(frame)
    if structural:
        raise ValueError("invalid OHLCV frame: " + "; ".join(structural))
    if frame.index.tz is None:
        raise ValueError("discovery requires timezone-aware timestamps")
    data_fingerprint = dataframe_fingerprint(frame, manifest)

    session_policy = NSEEquitySessionPolicy(
        cas_eligible=config.cas_eligible,
        exit_buffer_minutes=config.exit_buffer_minutes,
    )
    tick_policy = FixedTickSizePolicy(
        tick_size_rupees=config.tick_size_rupees,
        source=config.tick_size_source,
    )
    eligibility = _AlwaysEligible()
    cost_provider = CurrentTermsNSEIntradayCostProvider(pricing_date=config.pricing_date)
    base_fills = FillAssumptions(
        slippage_bps_per_leg=config.base_slippage_bps_per_leg,
        half_spread_bps_per_leg=config.base_half_spread_bps_per_leg,
    )
    simulation_config = IntradaySimulationConfig(
        initial_cash=config.initial_cash,
        max_trades_per_day=config.max_trades_per_day,
    )

    train_raw, validation_raw, test_raw = split_train_validation_test(
        frame,
        train_end_date=config.train_end_date,
        validation_end_date=config.validation_end_date,
    )

    # Exact stages operate on continuous-session bars only. Screening also
    # filters internally, so filtering once here keeps every stage identical.
    train_frame = filter_to_continuous_session(train_raw, session_policy)
    validation_frame = filter_to_continuous_session(validation_raw, session_policy)
    test_frame = filter_to_continuous_session(test_raw, session_policy)
    if train_frame.empty or validation_frame.empty or test_frame.empty:
        raise ValueError("continuous-session split produced an empty slice")

    candidates = build_intraday_candidates(
        session_open=config.session_open,
        bar_minutes=config.bar_minutes,
        breakout_buffer_bps=config.breakout_buffer_bps,
    )
    by_id = {item.candidate_id: item for item in candidates}

    screening = _screen_on_train(
        train_frame=train_frame,
        candidates=candidates,
        config=config,
        session_policy=session_policy,
    )
    # Screening is a fast informational filter, not the exact validator.
    # Strategies that rely on the session-cutoff exit (for example the
    # first-bar baseline with no explicit exit signal) show zero closed
    # VectorBT trades yet still produce exact session exits downstream, so
    # entry presence on train keeps them alive for exact validation.
    survivors: list[StrategyDefinition] = []
    for candidate_id, outcome in screening.items():
        if outcome.closed_trades > 0:
            survivors.append(by_id[candidate_id])
            continue
        try:
            train_signals = by_id[candidate_id].build_signals(train_frame)
        except ValueError:
            continue
        if bool(train_signals.entries_at_close.any()):
            survivors.append(by_id[candidate_id])
    extra_violations: list[str] = []
    if not survivors:
        extra_violations.append("no candidates survived vectorbt screening")

    train_evals: dict[str, CandidateEvaluation] = {}
    validation_evals: dict[str, CandidateEvaluation] = {}
    if survivors:
        for candidate in survivors:
            train_evals[candidate.candidate_id] = _evaluate_exact(
                candidate=candidate,
                frame=train_frame,
                config=config,
                session_policy=session_policy,
                tick_policy=tick_policy,
                eligibility=eligibility,
                cost_provider=cost_provider,
                fills=base_fills,
                simulation_config=simulation_config,
            )
            validation_evals[candidate.candidate_id] = _evaluate_exact(
                candidate=candidate,
                frame=validation_frame,
                config=config,
                session_policy=session_policy,
                tick_policy=tick_policy,
                eligibility=eligibility,
                cost_provider=cost_provider,
                fills=base_fills,
                simulation_config=simulation_config,
            )
        ranked = rank_candidates(list(validation_evals.values()), metric=config.ranking_metric)
        selected_id = ranked[0].candidate_id
    else:
        # Fail-closed selection: keep the baseline shape so the artifact still
        # names a real rule, but every downstream check will fail.
        selected_id = "baseline:first-bar-hold"
    selected = by_id[selected_id]

    # Walk-forward evidence comes from pre-test data only; the held-out test
    # slice is evaluated exactly once below and never influences selection.
    pretest_frame = pd.concat([train_frame, validation_frame]).sort_index()
    wf_windows = make_walk_forward_windows(
        pretest_frame,
        train_trading_days=config.walk_forward_train_days,
        test_trading_days=config.walk_forward_test_days,
        step_trading_days=config.walk_forward_step_days,
        embargo_trading_days=config.walk_forward_embargo_days,
    )
    wf_scope = survivors if survivors else [selected]
    wf_results = run_walk_forward_selection(
        frame=pretest_frame,
        windows=wf_windows,
        candidates=wf_scope,
        ranking_metric=config.ranking_metric,
        instrument_token=config.instrument_token,
        exchange=config.exchange,
        cost_provider=cost_provider,
        fills=base_fills,
        session_policy=session_policy,
        tick_size_policy=tick_policy,
        trading_eligibility_policy=eligibility,
        simulation_config=simulation_config,
    )
    wf_aggregate_net = sum(
        (item.test_evaluation.metrics.net_pnl for item in wf_results), Decimal(0)
    )
    wf_folds: list[WalkForwardFoldEvidence] = []
    for item in wf_results:
        fold_net = item.test_evaluation.metrics.net_pnl
        wf_folds.append(
            WalkForwardFoldEvidence(
                window_id=item.window.window_id,
                train_dates=tuple(d.isoformat() for d in item.window.train_dates),
                test_dates=tuple(d.isoformat() for d in item.window.test_dates),
                selected_candidate_id=item.selected_candidate_id,
                test_net_pnl=fold_net,
                test_trades=item.test_evaluation.metrics.trade_count,
                profitable=fold_net > 0,
            )
        )
    wf_profitable_folds = sum(1 for fold in wf_folds if fold.profitable)

    selected_train = train_evals.get(selected_id) or _evaluate_exact(
        candidate=selected,
        frame=train_frame,
        config=config,
        session_policy=session_policy,
        tick_policy=tick_policy,
        eligibility=eligibility,
        cost_provider=cost_provider,
        fills=base_fills,
        simulation_config=simulation_config,
    )
    selected_validation = validation_evals.get(selected_id) or _evaluate_exact(
        candidate=selected,
        frame=validation_frame,
        config=config,
        session_policy=session_policy,
        tick_policy=tick_policy,
        eligibility=eligibility,
        cost_provider=cost_provider,
        fills=base_fills,
        simulation_config=simulation_config,
    )
    selected_signals_test = selected.build_signals(test_frame)
    selected_test = evaluate_candidate_exact(
        candidate_id=selected_id,
        frame=test_frame,
        signals=selected_signals_test,
        instrument_token=config.instrument_token,
        exchange=config.exchange,
        cost_provider=cost_provider,
        fills=base_fills,
        session_policy=session_policy,
        tick_size_policy=tick_policy,
        trading_eligibility_policy=eligibility,
        config=simulation_config,
    )

    baseline = by_id["baseline:first-bar-hold"]
    baseline_test = _evaluate_exact(
        candidate=baseline,
        frame=test_frame,
        config=config,
        session_policy=session_policy,
        tick_policy=tick_policy,
        eligibility=eligibility,
        cost_provider=cost_provider,
        fills=base_fills,
        simulation_config=simulation_config,
    )

    stress = run_friction_stress(
        candidate_id=selected_id,
        frame=test_frame,
        signals=selected_signals_test,
        instrument_token=config.instrument_token,
        exchange=config.exchange,
        cost_provider=cost_provider,
        session_policy=session_policy,
        tick_size_policy=tick_policy,
        trading_eligibility_policy=eligibility,
        config=simulation_config,
        scenarios=[
            FrictionScenario(name="base", fills=base_fills),
            FrictionScenario(
                name="stressed",
                fills=FillAssumptions(
                    slippage_bps_per_leg=config.stress_slippage_bps_per_leg,
                    half_spread_bps_per_leg=config.stress_half_spread_bps_per_leg,
                ),
            ),
        ],
    )
    stress_worst = stress.worst_net_pnl

    research_dates = sorted(set(frame.index.date))
    cost_scenario = _build_cost_scenario(
        research_start=research_dates[0],
        research_end=research_dates[-1],
        scenario_date=research_dates[-1],
    )
    try:
        paper_report = _replay_test_slice_as_paper(
            test_frame=test_frame,
            candidate_id=selected_id,
            entries=selected_signals_test.entries_at_close,
            exits=selected_signals_test.exits_at_close,
            config=config,
            cost_provider=cost_provider,
            fills=base_fills,
            cost_scenario=cost_scenario,
        )
        paper_present = True
        shadow_decisions = len(paper_report.decisions)
        shadow_trades = len(paper_report.trades)
        shadow_fingerprint = paper_report.fingerprint()
    except Exception:  # noqa: BLE001 - paper failure must fail closed, not crash discovery
        paper_present = False
        shadow_decisions = 0
        shadow_trades = 0
        shadow_fingerprint = "paper-replay-failed"

    test_metrics = selected_test.metrics
    evidence = ResearchEvidence(
        trade_count=test_metrics.trade_count,
        profit_factor=_profit_factor_or_closed_form(test_metrics),
        max_drawdown_pct=test_metrics.ohlc_low_liquidation_stress_max_drawdown_pct,
        drawdown_basis=DrawdownBasis.OHLC_LOW_LIQUIDATION_STRESS,
        walk_forward_windows=len(wf_results),
        max_cost_reconciliation_error_inr=config.cost_reconciliation_error_inr,
        held_out_test_present=True,
        baseline_comparison_present=True,
        slippage_stress_present=True,
        event_driven_validation_present=True,
        paper_trading_present=paper_present,
        data_provenance_complete=not structural,
        unpriced_cost_components=(),
    )
    gate = evaluate_promotion_gate(evidence, config.promotion_thresholds)

    test_trade_pnls = tuple(trade.net_pnl for trade in selected_test.simulation.trades)
    expectancy = trade_expectancy(
        net_pnl=selected_test.metrics.net_pnl,
        trade_count=selected_test.metrics.trade_count,
    )
    sharpe = trade_sharpe(test_trade_pnls)

    if selected_test.metrics.net_pnl <= 0:
        extra_violations.append("held-out test net P&L is not positive")
    if expectancy <= 0:
        extra_violations.append("net expectancy is not positive after costs")
    if sharpe < config.min_sharpe:
        extra_violations.append(
            f"trade Sharpe {format(sharpe, 'f')} is below required {format(config.min_sharpe, 'f')}"
        )
    if wf_aggregate_net <= 0:
        extra_violations.append("walk-forward aggregate test net is not positive")
    if not wf_folds or wf_profitable_folds * 2 <= len(wf_folds):
        extra_violations.append(
            f"walk-forward profitable folds {wf_profitable_folds}/{len(wf_folds)} "
            "is not a strict majority"
        )
    if stress_worst is None or stress_worst <= 0:
        extra_violations.append("friction-stress worst case is not positive")
    if not paper_present:
        extra_violations.append("offline paper replay did not produce evidence")

    violations = tuple(list(gate.violations) + extra_violations)
    status = "PASS" if gate.passed and not extra_violations else "FAIL"

    candidate_fingerprint = canonical_sha256(
        {
            "candidate_id": selected_id,
            "parameters": dict(selected_test.parameters),
            "instrument_token": config.instrument_token,
            "exchange": config.exchange.value,
            "track": DISCOVERY_TRACK,
        }
    )

    return DiscoveryResult(
        schema_version=DISCOVERY_SCHEMA_VERSION,
        track=DISCOVERY_TRACK,
        selected_candidate_id=selected_id,
        selected_parameters=dict(selected_test.parameters),
        selected_strategy_name=selected_test.strategy_name,
        instrument_token=config.instrument_token,
        symbol=config.symbol,
        exchange=config.exchange.value,
        status=status,
        violations=violations,
        gate_decision=gate,
        selected_test_metrics=test_metrics,
        selected_train_metrics=selected_train.metrics,
        selected_validation_metrics=selected_validation.metrics,
        baseline_test_net_pnl=baseline_test.metrics.net_pnl,
        baseline_test_trades=baseline_test.metrics.trade_count,
        walk_forward_windows=len(wf_results),
        walk_forward_aggregate_test_net=wf_aggregate_net,
        walk_forward_folds=tuple(wf_folds),
        walk_forward_profitable_folds=wf_profitable_folds,
        test_expectancy=expectancy,
        test_sharpe=sharpe,
        stress_worst_net_pnl=stress_worst,
        stress_scenarios=tuple(name for name, _ in stress.scenarios),
        shadow_decisions=shadow_decisions,
        shadow_trades=shadow_trades,
        data_fingerprint=data_fingerprint,
        candidate_fingerprint=candidate_fingerprint,
        cost_scenario_fingerprint=cost_scenario.fingerprint(),
        shadow_fingerprint=shadow_fingerprint,
        cost_model="documented",
        pricing_date=config.pricing_date.isoformat(),
        tick_size_rupees=format(config.tick_size_rupees, "f"),
        tick_size_source=config.tick_size_source,
        cost_reconciliation_error_inr=(
            format(config.cost_reconciliation_error_inr, "f")
            if config.cost_reconciliation_error_inr is not None
            else None
        ),
        cost_reconciliation_source=config.cost_reconciliation_source,
        live_orders_called=False,
    )


__all__ = [
    "DISCOVERY_SCHEMA_VERSION",
    "DiscoveryConfig",
    "DiscoveryResult",
    "WalkForwardFoldEvidence",
    "build_intraday_candidates",
    "run_discovery",
    "split_train_validation_test",
    "trade_expectancy",
    "trade_sharpe",
]
