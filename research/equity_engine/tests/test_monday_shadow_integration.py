"""Cross-contract proof for Monday's research-to-shadow staging path.

This test keeps the path synthetic and local:

Kiro acquisition -> validated continuous dataset -> research simulator
-> theoretical shadow session.

No paper-trading artifact is required to start the first shadow session, and no
broker client or live-order surface is introduced by this integration.
"""

from __future__ import annotations

import hashlib
import inspect
from datetime import date, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
import pytest

from equity_engine.cost_ledger import EffectiveDatedCostLedger, LedgerComponent, LedgerProduct
from equity_engine.documented_costs import CurrentTermsNSEIntradayCostProvider
from equity_engine.event_simulator import (
    FillAssumptions,
    IntradaySimulationConfig,
    simulate_long_intraday,
)
from equity_engine.experiment import ApprovedCapital
from equity_engine.historical_cost_scenario import (
    MissingScenarioAssumptionError,
    ScenarioAssumption,
    compile_historical_scenario,
)
from equity_engine.historical_membership import (
    HistoricalTradingEligibilityPolicy,
    HistoricalTradingStatus,
    assess_historical_membership,
)
from equity_engine.historical_validation import IntradaySessionRule
from equity_engine.market_sessions import NSEEquitySessionPolicy
from equity_engine.models import Exchange
from equity_engine.shadow_execution import (
    AlwaysSignalShadowStrategy,
    ShadowEngineConfig,
    ShadowInstrumentIdentity,
    ShadowMarketEvent,
    ShadowRejectReason,
    ShadowSessionEngine,
)
from equity_engine.tick_size import FixedTickSizePolicy
from equity_engine.upstox_batch_history import (
    HistoricalAcquisitionEvidence,
    HistoricalBatchCandidate,
    UpstoxHistoricalBatchDownloader,
)
from equity_engine.validated_dataset_handoff import (
    RawAcquisitionArtifact,
    build_validated_dataset_handoff,
    session_policy_fingerprint,
)

TRADE_DATE = date(2026, 9, 7)
INSTRUMENT = "NSE_EQ|INE001A01036"
IST = ZoneInfo("Asia/Kolkata")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


class _SyntheticKiroClient:
    """GET-only Kiro-shaped acquisition source; it never leaves this process."""

    def get(self, url: str, **_: object) -> httpx.Response:
        index = pd.date_range(
            f"{TRADE_DATE.isoformat()} 09:15",
            periods=72,
            freq="5min",
            tz="Asia/Kolkata",
        ).append(
            pd.date_range(
                f"{TRADE_DATE.isoformat()} 15:15",
                periods=4,
                freq="5min",
                tz="Asia/Kolkata",
            )
        )
        candles = [
            [timestamp.isoformat(), 100, 101, 99, 100.5, 1000 + i, 0]
            for i, timestamp in enumerate(index)
        ]
        return httpx.Response(
            200,
            json={"status": "success", "data": {"candles": candles}},
            request=httpx.Request("GET", url),
        )


def _session_rule() -> IntradaySessionRule:
    return IntradaySessionRule(
        rule_id="synthetic-monday-cas-session",
        timezone="Asia/Kolkata",
        start_time=time(9, 15),
        end_time=time(15, 15),
        interval_minutes=5,
        source_reference="synthetic-monday-session-evidence",
        auxiliary_start_time=time(15, 15),
        auxiliary_end_time=time(15, 35),
        auxiliary_semantics="synthetic-CAS-auxiliary",
    )


def _historical_scenario():
    def assumption(component: LedgerComponent, rate: str) -> ScenarioAssumption:
        return ScenarioAssumption(
            assumption_id=f"monday-{component.value}",
            component=component,
            product=LedgerProduct.INTRADAY,
            basis="synthetic staging assumption",
            rate=Decimal(rate),
            formula=f"rate-{rate}",
            source="synthetic staging fixture",
            reason="account-specific historical actual unavailable",
        )

    return compile_historical_scenario(
        scenario_id="monday-shadow-staging-v1",
        ledger=EffectiveDatedCostLedger(),
        scenario_date=TRADE_DATE,
        research_start=TRADE_DATE,
        research_end=TRADE_DATE,
        product=LedgerProduct.INTRADAY,
        assumptions=(
            assumption(LedgerComponent.BROKERAGE, "0.0006"),
            assumption(LedgerComponent.GST, "0.18"),
            assumption(LedgerComponent.CLEARING, "0.000001"),
        ),
    )


def _acquire_and_validate(tmp_path: Path):
    policy = NSEEquitySessionPolicy(cas_eligible=True, exit_buffer_minutes=0)
    evidence = HistoricalAcquisitionEvidence(
        pit_fingerprint=_digest("monday:pit"),
        corporate_action_fingerprint=_digest("monday:corporate-actions"),
        acquisition_plan_fingerprint=_digest("monday:acquisition-plan"),
        session_policy_identity=session_policy_fingerprint(policy),
        expected_trade_dates=(TRADE_DATE,),
        session_rules={TRADE_DATE: _session_rule()},
    )
    result = UpstoxHistoricalBatchDownloader(
        access_token="synthetic-token-never-persisted",
        output_dir=tmp_path / "kiro-acquisition",
        client=_SyntheticKiroClient(),
        min_request_interval_seconds=0,
        max_attempts=1,
        backoff_seconds=0,
    ).run(
        candidates=[
            HistoricalBatchCandidate(
                instrument_key=INSTRUMENT,
                symbol="SYNTHETIC",
                start=TRADE_DATE,
                end=TRADE_DATE,
            )
        ],
        universe_rule_version="monday-pit-v1",
        adjustment_policy="raw-unadjusted",
        evidence=evidence,
    )
    assert result.passed is True
    manifest_path = Path(result.items[0].manifest)
    artifact = RawAcquisitionArtifact.from_kiro_manifest(manifest_path)
    return (
        artifact,
        policy,
        build_validated_dataset_handoff(
            artifact,
            expected_instrument_key=INSTRUMENT,
            expected_interval="5m",
            session_policy=policy,
        ),
    )


def _event(number: int, row: pd.Series) -> ShadowMarketEvent:
    timestamp = row.name.to_pydatetime()
    values = {name: Decimal(str(row[name])) for name in ("open", "high", "low", "close")}
    return ShadowMarketEvent(
        event_id=f"monday-event-{number}",
        seq=number,
        instrument_key=INSTRUMENT,
        exchange=Exchange.NSE,
        bar_timestamp=timestamp,
        bar_open=values["open"],
        bar_high=values["high"],
        bar_low=values["low"],
        bar_close=values["close"],
        bar_volume=int(row["volume"]),
        received_at=timestamp + timedelta(seconds=5),
        is_cas_auxiliary=False,
        feed_connected=True,
    )


def test_acquisition_research_output_configures_first_shadow_session(tmp_path: Path) -> None:
    """Prove the complete staging handoff and the theoretical no-order boundary."""
    artifact, session_policy, handoff = _acquire_and_validate(tmp_path)
    _, descriptor, research_input = handoff
    frame = research_input.frame_for_research()

    assert artifact.manifest.status.value == "COMPLETE"
    assert descriptor.raw_row_count == 76
    assert descriptor.research_row_count == 72
    assert descriptor.excluded_auxiliary_row_count == 4
    assert all(timestamp.time() < time(15, 15) for timestamp in frame.index)
    assert research_input.fingerprint == descriptor.deterministic_fingerprint()

    # Validated continuous data is the only frame accepted by research execution.
    entries = pd.Series(False, index=frame.index)
    exits = pd.Series(False, index=frame.index)
    entries.iloc[0] = True
    exits.iloc[1] = True
    eligibility = HistoricalTradingEligibilityPolicy(
        assess_historical_membership(
            instrument_key=INSTRUMENT,
            trading_dates=(TRADE_DATE,),
            statuses=(
                HistoricalTradingStatus(
                    trade_date=TRADE_DATE,
                    instrument_key=INSTRUMENT,
                    listed_on_nse=True,
                    normal_equity=True,
                    tradeable_in_normal_market=True,
                    source="synthetic-monday-membership",
                ),
            ),
        )
    )
    scenario = _historical_scenario()
    research_result = simulate_long_intraday(
        frame=frame,
        entries_at_close=entries,
        exits_at_close=exits,
        instrument_token=INSTRUMENT,
        exchange=Exchange.NSE,
        cost_provider=CurrentTermsNSEIntradayCostProvider(pricing_date=TRADE_DATE),
        fills=FillAssumptions(Decimal(1), Decimal("0.5")),
        session_policy=session_policy,
        tick_size_policy=FixedTickSizePolicy(Decimal("0.05"), "synthetic-tick-evidence"),
        trading_eligibility_policy=eligibility,
        config=IntradaySimulationConfig(Decimal(100000), 1),
    )
    assert len(research_result.trades) == 1

    # The descriptor is the research-output identity used to configure shadow.
    shadow = ShadowSessionEngine(
        instrument=ShadowInstrumentIdentity(
            instrument_key=descriptor.instrument_key,
            exchange=Exchange.NSE,
            cas_eligible=True,
            tick_size_rupees=Decimal("0.05"),
            source=f"validated-dataset:{descriptor.deterministic_fingerprint()}",
        ),
        strategy=AlwaysSignalShadowStrategy("research-output-strategy"),
        approved_capital=ApprovedCapital(amount_rupees=Decimal(100000), currency="INR"),
        cost_scenario=scenario,
        cost_provider=CurrentTermsNSEIntradayCostProvider(pricing_date=TRADE_DATE),
        fills=FillAssumptions(Decimal(1), Decimal("0.5")),
        config=ShadowEngineConfig(
            session_id=f"monday-shadow:{descriptor.deterministic_fingerprint()}",
            exit_buffer_minutes=0,
            max_quote_age_seconds=60,
            bar_interval_seconds=300,
            max_gap_multiplier=1.5,
            max_trades_per_session=1,
        ),
    )
    decisions = [shadow.process(_event(i, frame.iloc[i])) for i in range(2)]
    report = shadow.report()

    assert decisions[0].reason == ShadowRejectReason.ENTRY_SIGNAL_PENDING
    assert decisions[1].reason == ShadowRejectReason.THEORETICAL_ENTRY
    assert decisions[1].theoretical_entry is not None
    assert report.live_orders_called is False
    assert all(decision.live_orders_called is False for decision in decisions)
    assert not hasattr(shadow, "place_order")
    assert "paper_trading" not in inspect.signature(ShadowSessionEngine).parameters

    shadow_source = Path(__file__).parents[1] / "src" / "equity_engine" / "shadow_execution.py"
    source = shadow_source.read_text(encoding="utf-8")
    assert "import httpx" not in source
    assert "import requests" not in source
    assert all(name not in source for name in ("place_order", "modify_order", "cancel_order"))


def test_staging_cost_contract_is_historical_scenario_and_unknown_is_not_zero() -> None:
    scenario = _historical_scenario()
    assert scenario.historical_actual is False
    assert scenario.unknowns

    with pytest.raises(MissingScenarioAssumptionError, match="unknown is never zero"):
        compile_historical_scenario(
            scenario_id="monday-missing-cost",
            ledger=EffectiveDatedCostLedger(),
            scenario_date=TRADE_DATE,
            research_start=TRADE_DATE,
            research_end=TRADE_DATE,
            product=LedgerProduct.INTRADAY,
            assumptions=(),
        )


def test_shadow_does_not_require_paper_evidence_to_start(tmp_path: Path) -> None:
    _, _, (_, descriptor, research_input) = _acquire_and_validate(tmp_path)
    first = research_input.frame_for_research().iloc[0]
    engine = ShadowSessionEngine(
        instrument=ShadowInstrumentIdentity(
            instrument_key=descriptor.instrument_key,
            exchange=Exchange.NSE,
            cas_eligible=True,
            tick_size_rupees=Decimal("0.05"),
            source="synthetic-validated-research",
        ),
        strategy=AlwaysSignalShadowStrategy("first-session-without-paper-evidence"),
        approved_capital=ApprovedCapital(amount_rupees=Decimal(100000), currency="INR"),
        cost_scenario=_historical_scenario(),
        cost_provider=CurrentTermsNSEIntradayCostProvider(pricing_date=TRADE_DATE),
        fills=FillAssumptions(Decimal(1), Decimal("0.5")),
        config=ShadowEngineConfig("monday-first-session", 0, 60, 300, 1.5, 1),
    )
    decision = engine.process(_event(0, first))

    assert decision.reason == ShadowRejectReason.ENTRY_SIGNAL_PENDING
    assert decision.live_orders_called is False
    assert engine.report().live_orders_called is False
