from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from .costs import CostProvider
from .event_simulator import FillAssumptions, IntradaySimulationConfig, SessionExitResolver
from .models import Exchange
from .strategies import StrategySignals
from .tournament import CandidateEvaluation, evaluate_candidate_exact


@dataclass(frozen=True)
class FrictionScenario:
    name: str
    fills: FillAssumptions

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("friction scenario name is required")


@dataclass(frozen=True)
class StressResult:
    candidate_id: str
    scenarios: tuple[tuple[str, CandidateEvaluation], ...]

    @property
    def worst_net_return_pct(self) -> Decimal:
        if not self.scenarios:
            raise ValueError("stress result contains no scenarios")
        return min(evaluation.metrics.net_return_pct for _, evaluation in self.scenarios)

    @property
    def worst_net_pnl(self) -> Decimal:
        if not self.scenarios:
            raise ValueError("stress result contains no scenarios")
        return min(evaluation.metrics.net_pnl for _, evaluation in self.scenarios)

    @property
    def all_scenarios_profitable(self) -> bool:
        return bool(self.scenarios) and all(
            evaluation.metrics.net_pnl > 0 for _, evaluation in self.scenarios
        )


def run_friction_stress(
    *,
    candidate_id: str,
    frame: pd.DataFrame,
    signals: StrategySignals,
    instrument_token: str,
    exchange: Exchange,
    cost_provider: CostProvider,
    session_policy: SessionExitResolver,
    config: IntradaySimulationConfig,
    scenarios: list[FrictionScenario],
) -> StressResult:
    """Evaluate identical signals under caller-supplied spread/slippage assumptions.

    No friction scenario is inferred. The research experiment must provide and document each one.
    This makes it impossible to silently switch from a realistic assumption to a favorable one
    after seeing the result.
    """

    if not scenarios:
        raise ValueError("at least one friction scenario is required")
    names = [scenario.name for scenario in scenarios]
    if len(set(names)) != len(names):
        raise ValueError("friction scenario names must be unique")

    evaluations: list[tuple[str, CandidateEvaluation]] = []
    for scenario in scenarios:
        evaluation = evaluate_candidate_exact(
            candidate_id=candidate_id,
            frame=frame,
            signals=signals,
            instrument_token=instrument_token,
            exchange=exchange,
            cost_provider=cost_provider,
            fills=scenario.fills,
            session_policy=session_policy,
            config=config,
        )
        evaluations.append((scenario.name, evaluation))

    return StressResult(candidate_id=candidate_id, scenarios=tuple(evaluations))
