import httpx

from equity_engine.upstox_market_context import UpstoxFullQuoteV3Client


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _Client:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def get(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return _Response(self.payload)


def test_build_market_snapshot_uses_prev_close_for_breadth() -> None:
    payload = {
        "status": "success",
        "data": {
            "NSE_INDEX:Nifty 50": {
                "instrument_token": "NSE_INDEX|Nifty 50",
                "timestamp": "2026-09-07T10:00:00+05:30",
                "last_price": 25200,
                "prev_close_price": 25000,
                "ohlc": {"open": 25100, "high": 25220, "low": 25080, "close": 25200},
            },
            "NSE_INDEX:India VIX": {
                "instrument_token": "NSE_INDEX|India VIX",
                "timestamp": "2026-09-07T10:00:00+05:30",
                "last_price": 16,
                "prev_close_price": 15,
                "ohlc": {"open": 15.5, "high": 16.2, "low": 15.4, "close": 16},
            },
            "NSE_EQ:AAA": {
                "instrument_token": "NSE_EQ|AAA",
                "timestamp": "2026-09-07T10:00:00+05:30",
                "last_price": 101,
                "prev_close_price": 100,
                "ohlc": {"open": 100, "high": 101, "low": 99, "close": 101},
            },
            "NSE_EQ:BBB": {
                "instrument_token": "NSE_EQ|BBB",
                "timestamp": "2026-09-07T10:00:00+05:30",
                "last_price": 99,
                "prev_close_price": 100,
                "ohlc": {"open": 100, "high": 101, "low": 98, "close": 99},
            },
            "NSE_EQ:CCC": {
                "instrument_token": "NSE_EQ|CCC",
                "timestamp": "2026-09-07T10:00:00+05:30",
                "last_price": 100,
                "prev_close_price": 100,
                "ohlc": {"open": 100, "high": 101, "low": 99, "close": 100},
            },
        },
    }
    fake = _Client(payload)
    provider = UpstoxFullQuoteV3Client(access_token="test-token", client=fake)

    snapshot = provider.build_market_snapshot(
        nifty_instrument_key="NSE_INDEX|Nifty 50",
        vix_instrument_key="NSE_INDEX|India VIX",
        breadth_instrument_keys=["NSE_EQ|AAA", "NSE_EQ|BBB", "NSE_EQ|CCC"],
    )

    assert snapshot.advancers == 1
    assert snapshot.decliners == 1
    assert snapshot.unchanged == 1
    assert str(snapshot.nifty_last) == "25200"
    assert len(fake.calls) == 1


def test_missing_requested_quote_fails_closed() -> None:
    fake = _Client(
        {
            "status": "success",
            "data": {
                "NSE_INDEX:Nifty 50": {
                    "instrument_token": "NSE_INDEX|Nifty 50",
                    "timestamp": "2026-09-07T10:00:00+05:30",
                    "last_price": 25200,
                    "prev_close_price": 25000,
                    "ohlc": {"open": 25100},
                }
            },
        }
    )
    provider = UpstoxFullQuoteV3Client(access_token="test-token", client=fake)

    try:
        provider.fetch_by_instrument_token(["NSE_INDEX|Nifty 50", "NSE_INDEX|India VIX"])
    except RuntimeError as exc:
        assert "did not return requested instruments" in str(exc)
    else:
        raise AssertionError("missing quote should fail closed")


def test_partial_quote_batches_are_sorted_and_capped_at_500() -> None:
    requested = [f"NSE_EQ|INE{i:09d}" for i in range(1001)]
    calls: list[list[str]] = []

    def handler(request):
        keys = request.url.params["instrument_key"].split(",")
        calls.append(keys)
        return httpx.Response(
            200,
            json={
                "status": "success",
                "data": {
                    f"NSE_EQ:{key.split('|', 1)[1]}": {
                        "instrument_token": key,
                        "last_price": 100,
                        "timestamp": "2026-09-09T10:00:00+05:30",
                        "cas_eligible": False,
                    }
                    for key in keys
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = UpstoxFullQuoteV3Client(access_token="secret-token", client=client).fetch_partial_by_instrument_token(
        list(reversed(requested))
    )

    assert result.request_count == 3
    assert [len(chunk) for chunk in calls] == [500, 500, 1]
    assert calls[0] == sorted(calls[0])
    assert len(result.quotes) == len(requested)
    assert not result.failures


def test_partial_quote_result_preserves_missing_quote_reason_without_token() -> None:
    requested = ["NSE_EQ|INE000000001", "NSE_EQ|INE000000002"]
    fake = _Client(
        {
            "status": "success",
            "data": {
                "first": {
                    "instrument_token": requested[0],
                    "last_price": 100,
                    "timestamp": "2026-09-09T10:00:00+05:30",
                    "cas_eligible": False,
                }
            },
        }
    )

    result = UpstoxFullQuoteV3Client(
        access_token="secret-token", client=fake
    ).fetch_partial_by_instrument_token(requested)

    assert result.failures == {requested[1]: "missing_quote"}
    assert "secret-token" not in repr(result)
