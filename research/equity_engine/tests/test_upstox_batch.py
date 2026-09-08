import json
from datetime import UTC, date, datetime
from decimal import Decimal

import httpx
import pandas as pd
import pytest

from equity_engine.provenance import MarketDataManifest, dataframe_fingerprint
from equity_engine.upstox_batch import UpstoxBatchTask, UpstoxHistoricalBatchAcquirer
from equity_engine.upstox_history import HistoricalDataset, UpstoxHistoricalDataProvider


def _provider_with_retry(sleeps: list[float]) -> UpstoxHistoricalDataProvider:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(
            200,
            json={
                "status": "success",
                "data": {"candles": [["2024-07-01T09:15:00+05:30", 100, 101, 99, 100.5, 1000, 0]]},
            },
            request=request,
        )

    return UpstoxHistoricalDataProvider(
        access_token="secret-token",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=1,
        retry_backoff_seconds=0.25,
        sleep=sleeps.append,
    )


def test_upstox_history_retries_transient_http_failure_without_logging_token() -> None:
    sleeps: list[float] = []
    provider = _provider_with_retry(sleeps)
    dataset = provider.fetch_minutes(
        instrument_token="NSE_EQ|INE001A01036",
        symbol="OPEN",
        exchange="NSE",
        start=date(2024, 7, 1),
        end=date(2024, 7, 1),
        interval_minutes=5,
        universe_rule_version="test",
        adjustment_policy="no-fill",
    )

    assert len(dataset.frame) == 1
    assert sleeps == [0.25]
    assert "secret-token" not in repr(dataset)


def _dataset() -> HistoricalDataset:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-07-01 09:15:00", tz="Asia/Kolkata"),
            pd.Timestamp("2024-07-02 09:15:00", tz="Asia/Kolkata"),
        ],
        name="timestamp",
    )
    frame = pd.DataFrame(
        {
            "open": [100, 101],
            "high": [101, 102],
            "low": [99, 100],
            "close": [100.5, 101.5],
            "volume": [1000, 1100],
            "open_interest": [0, 0],
        },
        index=index,
    )
    manifest = MarketDataManifest(
        provider="upstox_v3",
        exchange="NSE",
        instrument_token="NSE_EQ|INE001A01036",
        symbol="OPEN",
        timezone="Asia/Kolkata",
        interval="5m",
        timestamp_semantics="candle_start",
        start=index[0].to_pydatetime(),
        end=index[-1].to_pydatetime(),
        retrieved_at=datetime(2026, 9, 7, tzinfo=UTC),
        adjustment_policy="no-fill",
        universe_rule_version="test",
        source_reference="test",
    )
    return HistoricalDataset(
        frame=frame,
        manifest=manifest,
        fingerprint=dataframe_fingerprint(frame, manifest),
    )


def test_upstox_batch_filters_to_point_in_time_dates_without_forward_fill(tmp_path) -> None:
    acquirer = UpstoxHistoricalBatchAcquirer(
        provider=UpstoxHistoricalDataProvider(access_token="unused"),
        output_dir=tmp_path,
    )
    task = UpstoxBatchTask(
        instrument_key="NSE_EQ|INE001A01036",
        symbol="OPEN",
        isin="INE001A01036",
        start=date(2024, 7, 1),
        end=date(2024, 7, 2),
        eligible_dates=(date(2024, 7, 2),),
        reference_price_rupees=Decimal(100),
        affordable_quantity=9,
    )

    filtered = acquirer._filter_to_eligible_dates(_dataset(), task)
    assert len(filtered.frame) == 1
    assert filtered.frame.index[0].date() == date(2024, 7, 2)
    assert filtered.frame.iloc[0]["close"] == 101.5

    missing = task.__class__(**{**task.__dict__, "eligible_dates": (date(2024, 7, 3),)})
    with pytest.raises(ValueError, match="no candles for eligible date"):
        acquirer._filter_to_eligible_dates(_dataset(), missing)


def test_upstox_batch_plan_screens_before_any_api_call(tmp_path) -> None:
    audit = tmp_path / "daily" / "2024-07-01.json"
    audit.parent.mkdir()
    audit.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "instrument_key": "NSE_EQ|INE001A01036",
                        "symbol": "OPEN",
                        "isin": "INE001A01036",
                        "eligibility": True,
                    },
                    {
                        "instrument_key": "NSE_EQ|INE002A01034",
                        "symbol": "TOOEXPENSIVE",
                        "isin": "INE002A01034",
                        "eligibility": True,
                    },
                ]
            }
        )
    )
    manifest = tmp_path / "nse_universe_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "pipeline": "nse_historical_universe",
                "live_orders_called": False,
                "dates": {
                    "2024-07-01": {
                        "status": "complete",
                        "audit_file": "daily/2024-07-01.json",
                    }
                },
            }
        )
    )
    prices = tmp_path / "prices.json"
    prices.write_text(
        json.dumps(
            {
                "NSE_EQ|INE001A01036": "100",
                "NSE_EQ|INE002A01034": "5000",
            }
        )
    )
    acquirer = UpstoxHistoricalBatchAcquirer(
        provider=UpstoxHistoricalDataProvider(access_token="unused"),
        output_dir=tmp_path / "out",
    )

    plan = acquirer.plan_from_manifest(universe_manifest_path=manifest, price_file=prices)
    assert [task.instrument_key for task in plan.tasks] == ["NSE_EQ|INE001A01036"]
    assert plan.estimated_request_count == 1
    assert any(item["reason"] == "fails affordability filter" for item in plan.skipped)
