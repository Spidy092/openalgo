from decimal import Decimal

import httpx

from equity_engine.upstox_readiness import UpstoxReadinessProbe


def _client(*, primary_ip: str | None = "203.0.113.10") -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer test-token"

        if request.url.path == "/v2/user/profile":
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": {
                        "email": "must-not-leak@example.com",
                        "exchanges": ["NSE", "BSE"],
                        "products": ["D", "I"],
                    },
                },
            )
        if request.url.path == "/v3/user/get-funds-and-margin":
            assert request.headers.get("Api-Version") == "3.0"
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": {"available_to_trade": {"total": 1000.25}},
                },
            )
        if request.url.path == "/v2/user/ip":
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": {"primary_ip": primary_ip, "secondary_ip": None},
                },
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_readiness_passes_only_read_only_account_checks() -> None:
    with _client() as client:
        snapshot = UpstoxReadinessProbe(access_token="test-token", client=client).run()

    assert snapshot.passed
    assert snapshot.available_to_trade == Decimal("1000.25")
    assert snapshot.exchanges == ("NSE", "BSE")
    assert snapshot.products == ("D", "I")
    assert snapshot.primary_static_ip_configured


def test_missing_static_ip_blocks_live_readiness() -> None:
    with _client(primary_ip=None) as client:
        snapshot = UpstoxReadinessProbe(access_token="test-token", client=client).run()

    assert not snapshot.passed
    failed = {check.name for check in snapshot.checks if not check.passed}
    assert "primary_static_ip" in failed


def test_probe_never_exposes_profile_identity_fields() -> None:
    with _client() as client:
        snapshot = UpstoxReadinessProbe(access_token="test-token", client=client).run()

    serialized = repr(snapshot)
    assert "must-not-leak@example.com" not in serialized
    assert "test-token" not in serialized
