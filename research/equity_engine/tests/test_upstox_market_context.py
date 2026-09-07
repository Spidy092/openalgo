from equity_engine.upstox_market_context import UpstoxFullQuoteV3Client


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

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
