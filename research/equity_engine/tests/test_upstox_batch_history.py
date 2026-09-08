from datetime import date
import json
from pathlib import Path

import httpx

from equity_engine.upstox_batch_history import (
    HistoricalBatchCandidate,
    RateLimitedRetryClient,
    UpstoxHistoricalBatchDownloader,
    plan_historical_batch,
)


class _SequenceClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.calls = 0

    def get(self, url: str, **_: object) -> httpx.Response:
        response = self.responses[self.calls]
        self.calls += 1
        return response


class _HistoryClient:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, url: str, **_: object) -> httpx.Response:
        self.calls += 1
        payload = {
            "status": "success",
            "data": {
                "candles": [
                    ["2026-09-07T09:15:00+05:30", 100, 101, 99, 100.5, 1000, 0],
                    ["2026-09-07T09:20:00+05:30", 100.5, 102, 100, 101.5, 1200, 0],
                ]
            },
        }
        return httpx.Response(
            200,
            json=payload,
            request=httpx.Request("GET", url),
        )


class _NeverClient:
    def get(self, url: str, **_: object) -> httpx.Response:
        raise AssertionError(f"network should not be called during resume: {url}")


def _candidate() -> HistoricalBatchCandidate:
    return HistoricalBatchCandidate(
        instrument_key="NSE_EQ|INE001A01036",
        symbol="OPEN",
        start=date(2026, 9, 7),
        end=date(2026, 9, 7),
    )


def test_plan_reports_request_and_storage_estimates() -> None:
    candidate = HistoricalBatchCandidate(
        instrument_key="NSE_EQ|INE001A01036",
        symbol="OPEN",
        start=date(2026, 1, 1),
        end=date(2026, 2, 28),
    )
    plan = plan_historical_batch(
        candidates=[candidate],
        interval_minutes=5,
        expected_rows_per_trading_day=75,
        estimated_bytes_per_row=80,
        trading_day_counts={candidate.instrument_key: 40},
        affordability_prefilter_applied=False,
    )
    assert plan.estimated_requests == 3
    assert plan.estimated_rows == 3000
    assert plan.estimated_storage_bytes == 240000
    assert plan.affordability_prefilter_applied is False


def test_retry_client_retries_transient_status() -> None:
    responses = [
        httpx.Response(429, request=httpx.Request("GET", "https://example.invalid")),
        httpx.Response(200, request=httpx.Request("GET", "https://example.invalid")),
    ]
    inner = _SequenceClient(responses)
    clock = iter([0.0, 0.0, 1.0])
    client = RateLimitedRetryClient(
        inner=inner,
        min_interval_seconds=0,
        max_attempts=2,
        backoff_seconds=0,
        sleep=lambda _: None,
        monotonic=lambda: next(clock),
    )
    assert client.get("https://example.invalid").status_code == 200
    assert inner.calls == 2


def test_downloader_saves_manifest_without_token_and_resumes(tmp_path: Path) -> None:
    token = "super-secret-token"
    network = _HistoryClient()
    first = UpstoxHistoricalBatchDownloader(
        access_token=token,
        output_dir=tmp_path,
        client=network,
        min_request_interval_seconds=0,
        sleep=lambda _: None,
    ).run(
        candidates=[_candidate()],
        universe_rule_version="test-rule",
        adjustment_policy="raw",
    )
    assert first.passed is True
    assert network.calls == 1
    item = first.items[0]
    manifest_text = Path(item.manifest).read_text(encoding="utf-8")
    assert token not in manifest_text
    assert json.loads(manifest_text)["live_orders_called"] is False

    second = UpstoxHistoricalBatchDownloader(
        access_token=token,
        output_dir=tmp_path,
        client=_NeverClient(),
        min_request_interval_seconds=0,
        sleep=lambda _: None,
    ).run(
        candidates=[_candidate()],
        universe_rule_version="test-rule",
        adjustment_policy="raw",
    )
    assert second.passed is True
    assert second.items[0].retrieval == "cached"
