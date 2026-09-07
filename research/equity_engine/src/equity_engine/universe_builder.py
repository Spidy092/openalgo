from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Iterable

import pandas as pd

from .costs import CostProvider
from .historical_membership import (
    HistoricalMembershipAssessment,
    filter_frame_to_eligible_dates,
)
from .liquidity import summarize_historical_liquidity
from .models import Exchange, Product
from .sizing import max_affordable_buy_quantity
from .tick_size import TickPolicy, assess_tick_policy_coverage
from .universe import (
    CorporateActionAssessment,
    ResearchUniverseThresholds,
    UniverseDecision,
    evaluate_research_universe_candidate,
)


@dataclass(frozen=True)
class ResearchUniverseCandidateData:
    instrument_key: str
    frame: pd.DataFrame
    dataset_fingerprint: str
    historical_membership: HistoricalMembershipAssessment
    tick_size_policy: TickPolicy
    corporate_actions: CorporateActionAssessment


@dataclass(frozen=True)
class ResearchUniverseAudit:
    instrument_key: str
    decision: UniverseDecision
    dataset_fingerprint: str
    data_start: pd.Timestamp | None
    data_end: pd.Timestamp | None
    eligible_trading_days: int
    affordable_quantity: int | None
    last_price_rupees: Decimal | None


@dataclass(frozen=True)
class ResearchUniverseBuildResult:
    selection_cutoff: date
    audits: tuple[ResearchUniverseAudit, ...]

    @property
    def eligible_instrument_keys(self) -> tuple[str, ...]:
        return tuple(
            audit.instrument_key for audit in self.audits if audit.decision.eligible
        )

    @property
    def rejected_instrument_keys(self) -> tuple[str, ...]:
        return tuple(
            audit.instrument_key for audit in self.audits if not audit.decision.eligible
        )


def _reject(
    *,
    candidate: ResearchUniverseCandidateData,
    violations: list[str],
) -> ResearchUniverseAudit:
    return ResearchUniverseAudit(
        instrument_key=candidate.instrument_key,
        decision=UniverseDecision(
            instrument_key=candidate.instrument_key,
            eligible=False,
            violations=tuple(violations),
        ),
        dataset_fingerprint=candidate.dataset_fingerprint,
        data_start=None if candidate.frame.empty else candidate.frame.index[0],
        data_end=None if candidate.frame.empty else candidate.frame.index[-1],
        eligible_trading_days=0,
        affordable_quantity=None,
        last_price_rupees=None,
    )


def build_research_universe(
    *,
    candidates: Iterable[ResearchUniverseCandidateData],
    selection_cutoff: date,
    capital_rupees: Decimal,
    thresholds: ResearchUniverseThresholds,
    cost_provider: CostProvider,
) -> ResearchUniverseBuildResult:
    """Build a historical research universe using only data available by `selection_cutoff`.

    The caller controls the lookback window by the frames it supplies. Any frame containing data
    after the cutoff is rejected, preventing universe construction from peeking into a held-out
    test/future period. Current Upstox MIS/suspension state is intentionally absent here.
    """

    if capital_rupees <= 0:
        raise ValueError("capital_rupees must be positive")

    candidate_list = list(candidates)
    keys = [candidate.instrument_key for candidate in candidate_list]
    if len(set(keys)) != len(keys):
        raise ValueError("research universe candidate instrument keys must be unique")

    audits: list[ResearchUniverseAudit] = []
    for candidate in candidate_list:
        if not candidate.instrument_key:
            raise ValueError("candidate instrument_key is required")
        if not candidate.dataset_fingerprint.strip():
            audits.append(_reject(candidate=candidate, violations=["dataset fingerprint is missing"]))
            continue
        if candidate.frame.empty:
            audits.append(_reject(candidate=candidate, violations=["historical OHLCV frame is empty"]))
            continue
        if candidate.frame.index.tz is None:
            audits.append(
                _reject(candidate=candidate, violations=["historical OHLCV timestamps are timezone-naive"])
            )
            continue
        if candidate.frame.index.max().date() > selection_cutoff:
            audits.append(
                _reject(
                    candidate=candidate,
                    violations=[
                        "universe lookahead detected: OHLCV extends beyond selection cutoff"
                    ],
                )
            )
            continue
        if candidate.historical_membership.instrument_key != candidate.instrument_key:
            audits.append(
                _reject(
                    candidate=candidate,
                    violations=["historical membership belongs to a different instrument"],
                )
            )
            continue
        if not candidate.historical_membership.complete:
            audits.append(
                _reject(
                    candidate=candidate,
                    violations=[
                        "point-in-time exchange evidence is incomplete for universe selection"
                    ],
                )
            )
            continue

        eligible_frame = filter_frame_to_eligible_dates(
            candidate.frame,
            candidate.historical_membership,
        )
        if eligible_frame.empty:
            audits.append(
                _reject(
                    candidate=candidate,
                    violations=["no exchange-eligible OHLCV rows remain after point-in-time filtering"],
                )
            )
            continue

        eligible_dates = tuple(sorted(set(eligible_frame.index.date)))
        tick_coverage = assess_tick_policy_coverage(
            policy=candidate.tick_size_policy,
            trading_dates=eligible_dates,
        )

        last_price = Decimal(str(eligible_frame.iloc[-1]["close"]))
        affordability = max_affordable_buy_quantity(
            instrument_token=candidate.instrument_key,
            exchange=Exchange.NSE,
            product=Product.INTRADAY,
            price=last_price,
            cash_limit=capital_rupees,
            cost_provider=cost_provider,
        )
        liquidity = summarize_historical_liquidity(
            frame=eligible_frame,
            affordable_quantity_after_entry_costs=affordability.quantity,
        )
        decision = evaluate_research_universe_candidate(
            instrument_key=candidate.instrument_key,
            liquidity=liquidity,
            corporate_actions=candidate.corporate_actions,
            historical_membership=candidate.historical_membership,
            tick_coverage=tick_coverage,
            thresholds=thresholds,
        )
        audits.append(
            ResearchUniverseAudit(
                instrument_key=candidate.instrument_key,
                decision=decision,
                dataset_fingerprint=candidate.dataset_fingerprint,
                data_start=eligible_frame.index[0],
                data_end=eligible_frame.index[-1],
                eligible_trading_days=len(eligible_dates),
                affordable_quantity=affordability.quantity,
                last_price_rupees=last_price,
            )
        )

    return ResearchUniverseBuildResult(
        selection_cutoff=selection_cutoff,
        audits=tuple(sorted(audits, key=lambda item: item.instrument_key)),
    )
