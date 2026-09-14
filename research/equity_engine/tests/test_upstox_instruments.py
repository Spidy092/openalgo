import gzip
import hashlib
import json

import httpx
import pytest

from equity_engine.upstox_instruments import UpstoxPublicInstrumentFiles


def test_public_instrument_loader_decodes_gzip_and_keeps_headers() -> None:
    payload = [{"instrument_key": "NSE_EQ|INE000000001", "segment": "NSE_EQ"}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=gzip.compress(json.dumps(payload).encode("utf-8")),
            headers={"etag": '"abc"', "last-modified": "Mon, 07 Sep 2026 00:00:00 GMT"},
        )

    provider = UpstoxPublicInstrumentFiles(client=httpx.Client(transport=httpx.MockTransport(handler)))
    result = provider.fetch("https://example.test/NSE.json.gz")

    assert result.rows[0]["instrument_key"] == "NSE_EQ|INE000000001"
    expected_payload = gzip.compress(json.dumps(payload).encode("utf-8"))
    assert result.sha256 == hashlib.sha256(expected_payload).hexdigest()
    assert result.etag == '"abc"'
    assert result.last_modified == "Mon, 07 Sep 2026 00:00:00 GMT"


def test_public_instrument_loader_rejects_non_gzip_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-gzip")

    provider = UpstoxPublicInstrumentFiles(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(ValueError, match="invalid Upstox gzip JSON"):
        provider.fetch("https://example.test/bad.gz")
