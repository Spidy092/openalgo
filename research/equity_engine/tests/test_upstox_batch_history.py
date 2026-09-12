import hashlib
import json
from datetime import date, time
from pathlib import Path

import httpx
import pandas as pd
import pytest

from equity_engine.historical_acquisition_plan import build_example_historical_acquisition_plan
from equity_engine.historical_validation import IntradaySessionRule
from equity_engine.market_sessions import NSEEquitySessionPolicy, filter_to_continuous_session
from equity_engine.upstox_batch_history import (
    HistoricalAcquisitionEvidence,
    HistoricalBatchCandidate,
    RateLimitedRetryClient,
    UpstoxHistoricalBatchDownloader,
    load_acquisition_evidence,
    load_acquisition_plan_binding,
    plan_historical_batch,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _rule(trade_date: date, *, cas: bool = False) -> IntradaySessionRule:
    return IntradaySessionRule(
        rule_id="test-cas-session" if cas else "test-session",
        timezone="Asia/Kolkata",
        start_time=time(9, 15),
        end_time=time(15, 15) if cas else time(9, 25),
        interval_minutes=5,
        source_reference="synthetic-nse-session-evidence",
        auxiliary_start_time=time(15, 15) if cas else None,
        auxiliary_end_time=time(15, 35) if cas else None,
        auxiliary_semantics="synthetic-CAS-auxiliary" if cas else None,
    )


def _evidence(
    dates: tuple[date, ...], *, label: str = "base", cas: bool = False
) -> HistoricalAcquisitionEvidence:
    return HistoricalAcquisitionEvidence(
        pit_fingerprint=_digest(f"{label}:pit"),
        corporate_action_fingerprint=_digest(f"{label}:corporate-actions"),
        acquisition_plan_fingerprint=_digest(f"{label}:plan"),
        session_policy_identity=f"{label}:session-policy",
        expected_trade_dates=dates,
        session_rules={day: _rule(day, cas=cas) for day in dates},
    )


def _payload_for_date(trade_date: date, *, rows: int = 2, cas: bool = False) -> dict[str, object]:
    index = pd.date_range(
        f"{trade_date.isoformat()} 09:15",
        periods=rows,
        freq="5min",
        tz="Asia/Kolkata",
    )
    if cas:
        rule = _rule(trade_date, cas=True)
        index = rule.expected_timestamps(trade_date).append(
            pd.date_range(
                f"{trade_date.isoformat()} 15:15",
                periods=4,
                freq="5min",
                tz="Asia/Kolkata",
            )
        )
    candles = [
        [timestamp.isoformat(), 100, 101, 99, 100.5, 1000 + number, 0]
        for number, timestamp in enumerate(index)
    ]
    return {"status": "success", "data": {"candles": candles}}


class _HistoryClient:
    def __init__(self, payloads: dict[date, dict[str, object]] | None = None) -> None:
        self.calls = 0
        self.payloads = payloads or {}

    def get(self, url: str, **_: object) -> httpx.Response:
        self.calls += 1
        chunk_start = date.fromisoformat(url.rsplit("/", 1)[-1])
        payload = self.payloads.get(chunk_start, _payload_for_date(chunk_start))
        return httpx.Response(
            200,
            json=payload,
            request=httpx.Request("GET", url),
        )


class _FailAfterFirstClient(_HistoryClient):
    def get(self, url: str, **kwargs: object) -> httpx.Response:
        if self.calls >= 1:
            self.calls += 1
            raise httpx.ConnectError("synthetic interruption", request=httpx.Request("GET", url))
        return super().get(url, **kwargs)


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


def _run(
    tmp_path: Path,
    client: object,
    *,
    candidate: HistoricalBatchCandidate | None = None,
    evidence: HistoricalAcquisitionEvidence | None = None,
    adjustment_policy: str = "raw-unadjusted",
):
    return UpstoxHistoricalBatchDownloader(
        access_token="synthetic-token-never-persisted",
        output_dir=tmp_path,
        client=client,  # type: ignore[arg-type]
        min_request_interval_seconds=0,
        max_attempts=4,
        backoff_seconds=0,
        sleep=lambda _: None,
    ).run(
        candidates=[candidate or _candidate()],
        universe_rule_version="test-rule",
        adjustment_policy=adjustment_policy,
        evidence=evidence or _evidence((_candidate().start,)),
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

    class SequenceClient:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, url: str, **_: object) -> httpx.Response:
            response = responses[self.calls]
            self.calls += 1
            return response

    inner = SequenceClient()
    client = RateLimitedRetryClient(
        inner=inner,
        min_interval_seconds=0,
        max_attempts=2,
        backoff_seconds=0,
        sleep=lambda _: None,
        monotonic=lambda: 0,
    )
    assert client.get("https://example.invalid").status_code == 200
    assert inner.calls == 2
    assert client.request_count == 2
    assert client.retry_count == 1


def test_immutable_raw_capture_complete_resume_and_credentials_not_persisted(
    tmp_path: Path,
) -> None:
    token = "super-secret-token"
    first = _run(tmp_path, _HistoryClient())
    assert first.passed is True
    assert first.requests == 1
    item = first.items[0]
    manifest_text = Path(item.manifest).read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["status"] == "COMPLETE"
    assert manifest["live_orders_called"] is False
    assert manifest["fingerprint_schema"] == "equity-market-data-v2"
    assert manifest["pit_evidence_fingerprint"] == _digest("base:pit")
    assert manifest["corporate_action_evidence_fingerprint"] == _digest("base:corporate-actions")
    assert manifest["acquisition_plan_fingerprint"] == _digest("base:plan")
    assert manifest["session_policy_identity"] == "base:session-policy"
    assert manifest["requested_start"] == "2026-09-07"
    assert manifest["covered_dates"] == ["2026-09-07"]
    raw_files = list(tmp_path.rglob("*.raw.json"))
    assert len(raw_files) == 1
    raw_before = raw_files[0].read_bytes()
    assert token not in raw_before.decode("utf-8")

    second = _run(tmp_path, _NeverClient())
    assert second.passed is True
    assert second.requests == 0
    assert second.items[0].retrieval == "cached"
    assert raw_files[0].read_bytes() == raw_before
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert token.encode() not in path.read_bytes()


def test_corrupt_cached_raw_sha_fails_closed_and_is_not_overwritten(tmp_path: Path) -> None:
    first = _run(tmp_path, _HistoryClient())
    raw_path = next(tmp_path.rglob("*.raw.json"))
    raw_path.write_bytes(raw_path.read_bytes() + b" ")

    second = _run(tmp_path, _NeverClient())
    assert second.passed is False
    assert any("SHA-256 mismatch" in failure for failure in second.failures)
    assert raw_path.read_bytes().endswith(b" ")
    state = json.loads(next(tmp_path.rglob("*.manifest.state.json")).read_text(encoding="utf-8"))
    assert state["status"] == "FAILED"
    assert first.items[0].state == "COMPLETE"


def test_partial_state_survives_restart_and_reuses_completed_chunk(tmp_path: Path) -> None:
    candidate = HistoricalBatchCandidate(
        instrument_key="NSE_EQ|INE001A01036",
        symbol="OPEN",
        start=date(2026, 9, 7),
        end=date(2026, 10, 5),
    )
    evidence = _evidence((date(2026, 9, 7), date(2026, 10, 5)))
    first = _run(tmp_path, _FailAfterFirstClient(), candidate=candidate, evidence=evidence)
    assert first.passed is False
    state_path = next(tmp_path.glob("**/*.manifest.state.json"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "PARTIAL"
    assert len(state["completed_chunks"]) == 1

    second = _run(
        tmp_path,
        _HistoryClient(
            {
                date(2026, 9, 7): _payload_for_date(date(2026, 9, 7)),
                date(2026, 10, 5): _payload_for_date(date(2026, 10, 5)),
            }
        ),
        candidate=candidate,
        evidence=evidence,
    )
    assert second.passed is True
    assert second.requests == 1
    assert second.items[0].retrieval == "downloaded"


def test_partial_cannot_become_complete_without_coverage_validation(tmp_path: Path) -> None:
    first = _run(
        tmp_path,
        _HistoryClient({date(2026, 9, 7): _payload_for_date(date(2026, 9, 7), rows=1)}),
    )
    assert first.passed is False
    assert not list(tmp_path.glob("**/*.parquet"))
    state = json.loads(next(tmp_path.glob("**/*.manifest.state.json")).read_text(encoding="utf-8"))
    assert state["status"] == "PARTIAL"

    second = _run(tmp_path, _NeverClient())
    assert second.passed is False
    assert not list(tmp_path.glob("**/*.parquet"))
    state = json.loads(next(tmp_path.glob("**/*.manifest.state.json")).read_text(encoding="utf-8"))
    assert state["status"] == "PARTIAL"


def test_duplicate_candidates_are_rejected_before_network() -> None:
    candidate = _candidate()
    downloader = UpstoxHistoricalBatchDownloader(
        access_token="synthetic-token",
        output_dir=Path("/tmp/unused-historical-pilot-test"),
        client=_NeverClient(),
        min_request_interval_seconds=0,
    )
    try:
        try:
            downloader.run(
                candidates=[candidate, candidate],
                universe_rule_version="test-rule",
                adjustment_policy="raw",
                evidence=_evidence((candidate.start,)),
            )
        except ValueError as exc:
            assert "unique" in str(exc)
        else:
            raise AssertionError("duplicate candidate was accepted")
    finally:
        downloader.close()


def test_manifest_identity_is_deterministic_and_binds_evidence(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = _run(first_dir, _HistoryClient(), evidence=_evidence((date(2026, 9, 7),)))
    second = _run(second_dir, _HistoryClient(), evidence=_evidence((date(2026, 9, 7),)))
    first_manifest = json.loads(Path(first.items[0].manifest).read_text(encoding="utf-8"))
    second_manifest = json.loads(Path(second.items[0].manifest).read_text(encoding="utf-8"))
    assert first_manifest["manifest_fingerprint"] == second_manifest["manifest_fingerprint"]
    assert first_manifest["data_fingerprint"] == second_manifest["data_fingerprint"]

    changed_dir = tmp_path / "changed"
    changed = _run(
        changed_dir,
        _HistoryClient(),
        evidence=_evidence((date(2026, 9, 7),), label="changed"),
    )
    changed_manifest = json.loads(Path(changed.items[0].manifest).read_text(encoding="utf-8"))
    assert changed_manifest["manifest_fingerprint"] != first_manifest["manifest_fingerprint"]


def test_cas_raw_rows_are_retained_but_continuous_path_excludes_auxiliary(tmp_path: Path) -> None:
    trade_date = date(2026, 9, 8)
    result = _run(
        tmp_path,
        _HistoryClient({trade_date: _payload_for_date(trade_date, cas=True)}),
        candidate=HistoricalBatchCandidate(
            instrument_key="NSE_EQ|INE001A01036",
            symbol="OPEN",
            start=trade_date,
            end=trade_date,
        ),
        evidence=_evidence((trade_date,), cas=True),
    )
    assert result.passed is True
    assert result.items[0].rows == 76
    assert result.items[0].continuous_session_rows == 72
    assert result.items[0].cas_auxiliary_rows == 4
    frame = pd.read_parquet(result.items[0].parquet)
    continuous = filter_to_continuous_session(
        frame,
        NSEEquitySessionPolicy(cas_eligible=True, exit_buffer_minutes=0),
    )
    assert len(continuous) == 72
    assert len(frame) == 76
    assert result.items[0].cas_auxiliary_rows > 0


def test_acquisition_evidence_json_round_trip_preserves_session_rules(tmp_path: Path) -> None:
    evidence = _evidence((date(2026, 9, 7),), cas=True)
    path = tmp_path / "acquisition-evidence.json"
    path.write_text(json.dumps(evidence.as_dict()), encoding="utf-8")

    loaded = load_acquisition_evidence(path)

    assert loaded.as_dict() == evidence.as_dict()
    assert loaded.session_rules[date(2026, 9, 7)].auxiliary_end_time == time(15, 35)


def test_canonical_acquisition_plan_binding_verifies_identity_and_intervals(
    tmp_path: Path,
) -> None:
    plan = build_example_historical_acquisition_plan()
    plan_path = tmp_path / "historical-acquisition-plan.json"
    plan_path.write_text(plan.to_json(), encoding="utf-8")
    binding = load_acquisition_plan_binding(plan_path)

    planned_intervals = tuple(item for item in binding.requested_intervals if item[1] == 5)
    expected_dates = tuple(sorted({day for item in planned_intervals for day in item[4]}))
    evidence = HistoricalAcquisitionEvidence(
        pit_fingerprint=binding.pit_evidence_fingerprint,
        corporate_action_fingerprint=binding.corporate_action_fingerprint,
        acquisition_plan_fingerprint=binding.deterministic_fingerprint,
        session_policy_identity="example-v2:nse-normal-sessions",
        expected_trade_dates=expected_dates,
        session_rules={day: _rule(day) for day in expected_dates},
    )
    candidates = tuple(
        HistoricalBatchCandidate(
            instrument_key=instrument_key,
            symbol=instrument_key.rsplit("|", 1)[-1],
            start=start,
            end=end,
        )
        for instrument_key, _, start, end, _ in planned_intervals
    )

    binding.validate_execution(
        candidates=candidates,
        interval_minutes=5,
        evidence=evidence,
    )
    tampered_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tampered_plan["frozen_wfo_population"] = ["NSE_EQ|tampered"]
    plan_path.write_text(json.dumps(tampered_plan), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_acquisition_plan_binding(plan_path)


def test_complete_resume_rejects_tampered_state_manifest_and_policy(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    _run(state_dir, _HistoryClient())
    state_path = next(state_dir.glob("**/*.manifest.state.json"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["adjustment_policy"] = "tampered-policy"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    state_result = _run(state_dir, _NeverClient())
    assert state_result.passed is False
    assert any("metadata mismatch" in failure for failure in state_result.failures)

    manifest_dir = tmp_path / "manifest"
    manifest_first = _run(manifest_dir, _HistoryClient())
    manifest_path = Path(manifest_first.items[0].manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rows"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_result = _run(manifest_dir, _NeverClient())
    assert manifest_result.passed is False
    assert any("metadata mismatch" in failure for failure in manifest_result.failures)

    policy_dir = tmp_path / "policy"
    _run(policy_dir, _HistoryClient())
    policy_result = _run(policy_dir, _NeverClient(), adjustment_policy="tampered-policy")
    assert policy_result.passed is False
    assert any("metadata mismatch" in failure for failure in policy_result.failures)


def test_batch_manifest_persists_actual_raw_byte_total(tmp_path: Path) -> None:
    result = _run(tmp_path, _HistoryClient())
    batch_manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    item_manifest = json.loads(Path(result.items[0].manifest).read_text(encoding="utf-8"))

    assert result.raw_bytes == item_manifest["raw_bytes"]
    assert batch_manifest["summary"]["raw_bytes"] == item_manifest["raw_bytes"]
    assert batch_manifest["summary"]["raw_bytes"] > 1
