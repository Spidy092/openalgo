from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import pandas as pd

from .costs import CostProvider
from .historical_membership import (
    HistoricalMembershipAssessment,
    filter_frame_to_eligible_dates,
)
from .liquidity import summarize_historical_liquidity
from .market_sessions import (
    ContinuousSessionPolicy,
    NSEEquitySessionPolicy,
    filter_to_continuous_session,
)
from .models import Exchange, Product
from .provenance import validate_ohlcv_frame
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
    session_policy: ContinuousSessionPolicy
    minimum_tradable_quantity: int
    minimum_tradable_quantity_source: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_tradable_quantity, bool)
            or not isinstance(self.minimum_tradable_quantity, int)
            or self.minimum_tradable_quantity <= 0
        ):
            raise ValueError("minimum_tradable_quantity must be a positive integer")
        if not self.minimum_tradable_quantity_source.strip():
            raise ValueError("minimum_tradable_quantity_source is required")


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
    approved_capital_rupees: Decimal | None = None
    reference_price_timestamp: pd.Timestamp | None = None
    minimum_tradable_quantity: int | None = None
    minimum_tradable_quantity_source: str | None = None
    required_capital_rupees: Decimal | None = None
    cash_required_rupees: Decimal | None = None
    entry_charges_rupees: Decimal | None = None
    liquidity_metric: str | None = None
    liquidity_value_rupees: Decimal | None = None
    liquidity_threshold_rupees: Decimal | None = None
    volume_value_shares: Decimal | None = None
    volume_threshold_shares: Decimal | None = None
    minimum_observed_trading_days: int | None = None
    liquidity_lookback_days: int | None = None
    session_policy_provenance: str | None = None
    cas_eligible: bool | None = None


@dataclass(frozen=True)
class ResearchUniverseBuildResult:
    selection_cutoff: date
    audits: tuple[ResearchUniverseAudit, ...]
    approved_capital_rupees: Decimal | None = None
    selection_as_of: pd.Timestamp | None = None
    reference_price_policy: str = "latest continuous-session close on or before selection as-of"

    def as_prefilter_artifact(self) -> dict[str, object]:
        """Return deterministic, JSON-ready evidence for the canonical prefilter result.

        The method deliberately does not write files or fetch data. A caller may persist this
        object with canonical JSON serialization alongside the candidate-file it approves.
        """

        def timestamp_text(value: pd.Timestamp | None) -> str | None:
            return value.isoformat() if value is not None else None

        def decimal_text(value: Decimal | None) -> str | None:
            return str(value) if value is not None else None

        return {
            "schema_version": "research-universe-prefilter-v1",
            "selection_cutoff": self.selection_cutoff.isoformat(),
            "selection_as_of": timestamp_text(self.selection_as_of),
            "approved_capital_rupees": decimal_text(self.approved_capital_rupees),
            "reference_price_policy": self.reference_price_policy,
            "candidates": [
                {
                    "instrument_key": audit.instrument_key,
                    "eligible": audit.decision.eligible,
                    "reason_codes": list(audit.decision.violations),
                    "dataset_fingerprint": audit.dataset_fingerprint,
                    "data_start": timestamp_text(audit.data_start),
                    "data_end": timestamp_text(audit.data_end),
                    "eligible_trading_days": audit.eligible_trading_days,
                    "reference_price_rupees": decimal_text(audit.last_price_rupees),
                    "reference_price_timestamp": timestamp_text(audit.reference_price_timestamp),
                    "minimum_tradable_quantity": audit.minimum_tradable_quantity,
                    "minimum_tradable_quantity_source": audit.minimum_tradable_quantity_source,
                    "required_capital_rupees": decimal_text(audit.required_capital_rupees),
                    "cash_required_rupees": decimal_text(audit.cash_required_rupees),
                    "entry_charges_rupees": decimal_text(audit.entry_charges_rupees),
                    "affordable_quantity": audit.affordable_quantity,
                    "liquidity_metric": audit.liquidity_metric,
                    "liquidity_value_rupees": decimal_text(audit.liquidity_value_rupees),
                    "liquidity_threshold_rupees": decimal_text(audit.liquidity_threshold_rupees),
                    "volume_value_shares": decimal_text(audit.volume_value_shares),
                    "volume_threshold_shares": decimal_text(audit.volume_threshold_shares),
                    "minimum_observed_trading_days": audit.minimum_observed_trading_days,
                    "liquidity_lookback_trading_days": audit.liquidity_lookback_days,
                    "session_policy_provenance": audit.session_policy_provenance,
                    "cas_eligible": audit.cas_eligible,
                }
                for audit in self.audits
            ],
            "live_orders_called": False,
        }

    @property
    def eligible_instrument_keys(self) -> tuple[str, ...]:
        return tuple(audit.instrument_key for audit in self.audits if audit.decision.eligible)

    @property
    def rejected_instrument_keys(self) -> tuple[str, ...]:
        return tuple(audit.instrument_key for audit in self.audits if not audit.decision.eligible)


def _reject(
    *,
    candidate: ResearchUniverseCandidateData,
    violations: list[str],
    approved_capital_rupees: Decimal | None = None,
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
        approved_capital_rupees=approved_capital_rupees,
        minimum_tradable_quantity=candidate.minimum_tradable_quantity,
        minimum_tradable_quantity_source=candidate.minimum_tradable_quantity_source,
        session_policy_provenance=_session_policy_provenance(candidate.session_policy),
        cas_eligible=getattr(candidate.session_policy, "cas_eligible", None),
    )


def _session_policy_provenance(policy: ContinuousSessionPolicy) -> str:
    """Return stable, non-market, non-volatile identity for the supplied session policy."""

    policy_type = f"{type(policy).__module__}.{type(policy).__qualname__}"
    if isinstance(policy, NSEEquitySessionPolicy):
        starts = tuple(
            (day.isoformat(), value.isoformat())
            for day, value in sorted(policy.special_session_continuous_start.items())
        )
        ends = tuple(
            (day.isoformat(), value.isoformat())
            for day, value in sorted(policy.special_session_continuous_end.items())
        )
        return (
            f"{policy_type}:cas_eligible={policy.cas_eligible}:"
            f"exit_buffer_minutes={policy.exit_buffer_minutes}:starts={starts}:ends={ends}"
        )
    return f"{policy_type}:{policy!r}"


def build_research_universe(
    *,
    candidates: Iterable[ResearchUniverseCandidateData],
    selection_cutoff: date,
    capital_rupees: Decimal,
    thresholds: ResearchUniverseThresholds,
    cost_provider: CostProvider,
    selection_as_of: pd.Timestamp | None = None,
) -> ResearchUniverseBuildResult:
    """Build a historical research universe using only data available by `selection_cutoff`.

    The caller controls the lookback window by the frames it supplies. Any frame containing data
    after the cutoff is rejected, preventing universe construction from peeking into a held-out
    test/future period. Current Upstox MIS/suspension state is intentionally absent here.
    ``capital_rupees`` is explicit approved algorithm capital. Affordability delegates to the
    canonical charge-aware sizing primitive; audit evidence also records the raw
    price-times-minimum-lot requirement. Broker balance is not consulted.
    """

    if (
        not isinstance(capital_rupees, Decimal)
        or not capital_rupees.is_finite()
        or capital_rupees <= 0
    ):
        raise ValueError("capital_rupees must be positive")
    if selection_as_of is not None:
        selection_as_of = pd.Timestamp(selection_as_of)
        if selection_as_of.tzinfo is None:
            raise ValueError("selection_as_of must be timezone-aware")
        if selection_as_of.date() > selection_cutoff:
            raise ValueError("selection_as_of cannot be after selection_cutoff")

    candidate_list = list(candidates)
    keys = [candidate.instrument_key for candidate in candidate_list]
    if len(set(keys)) != len(keys):
        raise ValueError("research universe candidate instrument keys must be unique")

    audits: list[ResearchUniverseAudit] = []
    for candidate in candidate_list:
        if not candidate.instrument_key:
            raise ValueError("candidate instrument_key is required")
        if not candidate.dataset_fingerprint.strip():
            audits.append(
                _reject(
                    candidate=candidate,
                    violations=["dataset fingerprint is missing"],
                    approved_capital_rupees=capital_rupees,
                )
            )
            continue
        if candidate.frame.empty:
            audits.append(
                _reject(
                    candidate=candidate,
                    violations=["historical OHLCV frame is empty"],
                    approved_capital_rupees=capital_rupees,
                )
            )
            continue
        if (
            not isinstance(candidate.frame.index, pd.DatetimeIndex)
            or candidate.frame.index.tz is None
        ):
            audits.append(
                _reject(
                    candidate=candidate,
                    violations=["historical OHLCV timestamps must be timezone-aware"],
                    approved_capital_rupees=capital_rupees,
                )
            )
            continue
        if candidate.frame.index.max().date() > selection_cutoff:
            audits.append(
                _reject(
                    candidate=candidate,
                    violations=[
                        "universe lookahead detected: OHLCV extends beyond selection cutoff"
                    ],
                    approved_capital_rupees=capital_rupees,
                )
            )
            continue
        if candidate.historical_membership.instrument_key != candidate.instrument_key:
            audits.append(
                _reject(
                    candidate=candidate,
                    violations=["historical membership belongs to a different instrument"],
                    approved_capital_rupees=capital_rupees,
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
                    approved_capital_rupees=capital_rupees,
                )
            )
            continue

        candidate_frame = candidate.frame
        if selection_as_of is not None:
            candidate_frame = candidate_frame.loc[candidate_frame.index <= selection_as_of]
        if candidate_frame.empty:
            audits.append(
                _reject(
                    candidate=candidate,
                    violations=["no OHLCV rows exist on or before selection_as_of"],
                    approved_capital_rupees=capital_rupees,
                )
            )
            continue
        try:
            frame_violations = validate_ohlcv_frame(candidate_frame)
        except (TypeError, ValueError) as exc:
            frame_violations = [f"OHLCV validation error: {type(exc).__name__}: {exc}"]
        if frame_violations:
            audits.append(
                _reject(
                    candidate=candidate,
                    violations=["invalid historical OHLCV: " + "; ".join(frame_violations)],
                    approved_capital_rupees=capital_rupees,
                )
            )
            continue

        eligible_frame = filter_frame_to_eligible_dates(
            candidate_frame, candidate.historical_membership
        )
        eligible_frame = filter_to_continuous_session(eligible_frame, candidate.session_policy)
        if eligible_frame.empty:
            audits.append(
                _reject(
                    candidate=candidate,
                    violations=[
                        "no exchange-eligible OHLCV rows remain after point-in-time filtering"
                    ],
                    approved_capital_rupees=capital_rupees,
                )
            )
            continue

        eligible_dates = tuple(sorted(set(eligible_frame.index.date)))
        tick_coverage = assess_tick_policy_coverage(
            policy=candidate.tick_size_policy,
            trading_dates=eligible_dates,
        )

        reference_price_timestamp = eligible_frame.index.max()
        last_price = Decimal(str(eligible_frame.loc[reference_price_timestamp, "close"]))
        if not last_price.is_finite() or last_price <= 0:
            audits.append(
                _reject(
                    candidate=candidate,
                    violations=["reference price is missing, non-finite, or non-positive"],
                    approved_capital_rupees=capital_rupees,
                )
            )
            continue
        affordability = max_affordable_buy_quantity(
            instrument_token=candidate.instrument_key,
            exchange=Exchange.NSE,
            product=Product.INTRADAY,
            price=last_price,
            cash_limit=capital_rupees,
            cost_provider=cost_provider,
            minimum_tradable_quantity=candidate.minimum_tradable_quantity,
        )
        liquidity = summarize_historical_liquidity(
            frame=eligible_frame,
            affordable_quantity_after_entry_costs=affordability.quantity,
            approved_capital_rupees=capital_rupees,
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
                approved_capital_rupees=capital_rupees,
                reference_price_timestamp=reference_price_timestamp,
                minimum_tradable_quantity=candidate.minimum_tradable_quantity,
                minimum_tradable_quantity_source=candidate.minimum_tradable_quantity_source,
                required_capital_rupees=(last_price * Decimal(candidate.minimum_tradable_quantity)),
                cash_required_rupees=affordability.cash_required,
                entry_charges_rupees=affordability.entry_charges,
                liquidity_metric="median daily close-volume notional proxy",
                liquidity_value_rupees=liquidity.median_daily_notional_proxy_rupees,
                liquidity_threshold_rupees=thresholds.min_median_daily_notional_proxy_rupees,
                volume_value_shares=liquidity.median_daily_volume_shares,
                volume_threshold_shares=thresholds.min_median_daily_volume_shares,
                minimum_observed_trading_days=thresholds.min_observed_trading_days,
                liquidity_lookback_days=len(eligible_dates),
                session_policy_provenance=_session_policy_provenance(candidate.session_policy),
                cas_eligible=getattr(candidate.session_policy, "cas_eligible", None),
            )
        )

    return ResearchUniverseBuildResult(
        selection_cutoff=selection_cutoff,
        audits=tuple(sorted(audits, key=lambda item: item.instrument_key)),
        approved_capital_rupees=capital_rupees,
        selection_as_of=selection_as_of,
    )
