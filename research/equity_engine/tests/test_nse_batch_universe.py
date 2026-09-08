import csv
import gzip
import io
from datetime import date
from pathlib import Path

import httpx

from equity_engine.nse_batch_universe import NseBatchUniverseBuilder
from equity_engine.nse_calendar import CalendarEvidence
from equity_engine.nse_semantics import (
    EffectiveDatedNseCmSemanticsPolicy,
    nse_cm_master_data_v15_semantics,
)

_HEADER = [
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


def _payload() -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_HEADER, lineterminator="\n")
    writer.writeheader()
    writer.writerow(
        {
            "FinInstrmId": "1",
            "TckrSymb": "OPEN",
            "SctySrs": "EQ",
            "FinInstrmNm": "OPEN LIMITED",
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


class _Client:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code
        self.calls = 0

    def get(self, url: str, **_: object) -> httpx.Response:
        self.calls += 1
        return httpx.Response(
            self.status_code,
            content=_payload() if self.status_code == 200 else b"",
            request=httpx.Request("GET", url),
        )


def _calendar(day: date) -> CalendarEvidence:
    return CalendarEvidence(
        trading_dates=(day,),
        holiday_dates=(),
        excluded_special_session_dates=(),
        source_urls=("https://example.invalid/nse-calendar",),
    )


def _builder(tmp_path: Path, client: _Client) -> NseBatchUniverseBuilder:
    return NseBatchUniverseBuilder(
        output_dir=tmp_path,
        semantics_policy=EffectiveDatedNseCmSemanticsPolicy([nse_cm_master_data_v15_semantics()]),
        client=client,
        sleep=lambda _: None,
    )


def test_batch_downloads_materializes_and_resumes_from_cache(tmp_path: Path) -> None:
    day = date(2026, 9, 7)
    client = _Client()
    first = _builder(tmp_path, client).build(
        calendar=_calendar(day),
        start=day,
        end=day,
    )
    assert first.passed is True
    assert first.unique_instruments == 1
    assert first.unique_eligible_instruments == 1
    assert client.calls == 1
    assert Path(first.days[0].universe_parquet).exists()

    second_client = _Client()
    second = _builder(tmp_path, second_client).build(
        calendar=_calendar(day),
        start=day,
        end=day,
    )
    assert second.passed is True
    assert second.days[0].retrieval == "cached"
    assert second_client.calls == 0


def test_batch_refuses_corrupt_cached_hash(tmp_path: Path) -> None:
    day = date(2026, 9, 7)
    client = _Client()
    first = _builder(tmp_path, client).build(
        calendar=_calendar(day),
        start=day,
        end=day,
    )
    assert first.passed
    raw = Path(first.days[0].raw_gzip)
    raw.with_suffix(raw.suffix + ".sha256").write_text("0" * 64 + "\n", encoding="utf-8")

    result = _builder(tmp_path, _Client()).build(
        calendar=_calendar(day),
        start=day,
        end=day,
    )
    assert result.passed is False
    assert "hash mismatch" in result.failures[0].error


def test_declared_trading_date_404_is_reported_as_failure(tmp_path: Path) -> None:
    day = date(2026, 9, 7)
    result = _builder(tmp_path, _Client(status_code=404)).build(
        calendar=_calendar(day),
        start=day,
        end=day,
    )
    assert result.passed is False
    assert result.days == ()
    assert "HTTP 404" in result.failures[0].error
