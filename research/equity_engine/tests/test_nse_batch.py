import csv
import gzip
import io
import json
from datetime import date

import httpx
import pytest

from equity_engine.nse_batch import (
    NseBatchAcquisitionError,
    NseCachePayloadConflictError,
    NseHistoricalUniverseBatch,
    NseMiiSnapshotDownloader,
)
from equity_engine.nse_trading_calendar import NseTradingCalendar

HEADER = [
    "FinInstrmId",
    "TckrSymb",
    "SctySrs",
    "FinInstrmNm",
    "ISIN",
    "NewBrdLotQty",
    "SctyTpFlg",
    "BidIntrvl",
    "CallAuctnInd",
    "PrtdToTrad",
    "SctyStsNrmlMkt",
    "ElgbltyNrmlMkt",
]


def _payload(symbol: str = "OPEN", *, day: date = date(2024, 7, 1)) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=HEADER, lineterminator="\n")
    writer.writeheader()
    writer.writerow(
        {
            "FinInstrmId": "1",
            "TckrSymb": symbol,
            "SctySrs": "EQ",
            "FinInstrmNm": f"{symbol} LIMITED",
            "ISIN": "INE001A01036",
            "NewBrdLotQty": "1",
            "SctyTpFlg": "0",
            "BidIntrvl": "5",
            "CallAuctnInd": "0",
            "PrtdToTrad": "1",
            "SctyStsNrmlMkt": "2",
            "ElgbltyNrmlMkt": "1",
        }
    )
    return gzip.compress(output.getvalue().encode("utf-8"))


def _calendar() -> NseTradingCalendar:
    return NseTradingCalendar(closed_dates=frozenset(), source_urls=("https://example.test",))


def test_snapshot_downloader_reuses_cache_and_hashes_raw_payload(tmp_path) -> None:
    calls = 0
    payload = _payload()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=payload, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    downloader = NseMiiSnapshotDownloader(cache_dir=tmp_path, client=client)
    first = downloader.acquire(date(2024, 7, 1))
    second = downloader.acquire(date(2024, 7, 1))

    assert first.status == "downloaded"
    assert second.status == "cached"
    assert calls == 1
    assert first.snapshot_sha256 == second.snapshot_sha256
    assert first.raw_path is not None and first.raw_path.exists()


def test_snapshot_downloader_never_overwrites_a_different_payload(tmp_path) -> None:
    original = _payload("ORIGINAL")
    replacement = _payload("REPLACEMENT")
    filename = "NSE_CM_security_01072024.csv.gz"
    (tmp_path / filename).write_bytes(original)

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=replacement, request=request)
        )
    )
    downloader = NseMiiSnapshotDownloader(cache_dir=tmp_path, client=client)

    with pytest.raises(NseCachePayloadConflictError):
        downloader.acquire(date(2024, 7, 1), refresh=True)
    assert (tmp_path / filename).read_bytes() == original


def test_batch_reports_missing_official_file_and_writes_fail_closed_manifest(tmp_path) -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(404, request=request))
    )
    batch = NseHistoricalUniverseBatch(
        output_dir=tmp_path,
        calendar=_calendar(),
        downloader=NseMiiSnapshotDownloader(cache_dir=tmp_path / "raw", client=client),
    )

    with pytest.raises(NseBatchAcquisitionError):
        batch.run(start=date(2024, 7, 1), end=date(2024, 7, 1))
    manifest = json.loads((tmp_path / "nse_universe_manifest.json").read_text())
    assert manifest["dates"]["2024-07-01"]["status"] == "missing"
    assert manifest["live_orders_called"] is False


def test_batch_materializes_audit_and_affordability_dry_run_report(tmp_path) -> None:
    payload = _payload()
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=payload, request=request)
        )
    )
    batch = NseHistoricalUniverseBatch(
        output_dir=tmp_path,
        calendar=_calendar(),
        downloader=NseMiiSnapshotDownloader(cache_dir=tmp_path / "raw", client=client),
    )

    result = batch.run(
        start=date(2024, 7, 1),
        end=date(2024, 7, 1),
        affordability_prices={"NSE_EQ|INE001A01036": 100},
    )
    manifest = result.manifest
    assert manifest["trading_dates"] == ["2024-07-01"]
    assert manifest["completed_dates"] == 1
    assert manifest["dry_run_screen"]["candidate_count_after_affordability_filter"] == 1
    assert manifest["dry_run_screen"]["estimated_api_request_count_5m"] == 1
    assert manifest["live_orders_called"] is False
    assert (tmp_path / "daily/2024-07-01.json").exists()
