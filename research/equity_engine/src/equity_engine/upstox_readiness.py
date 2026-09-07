from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx


UPSTOX_API_BASE = "https://api.upstox.com"


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class UpstoxReadinessSnapshot:
    checks: tuple[ReadinessCheck, ...]
    available_to_trade: Decimal | None
    exchanges: tuple[str, ...]
    products: tuple[str, ...]
    primary_static_ip_configured: bool
    secondary_static_ip_configured: bool

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


class UpstoxReadinessProbe:
    """Read-only verification of a funded Upstox account before any order code is enabled.

    This probe intentionally calls only profile, funds and static-IP endpoints. It never calls
    place/modify/cancel order APIs and it never prints or persists the bearer token.
    """

    def __init__(
        self,
        *,
        access_token: str,
        timeout_seconds: float = 20.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not access_token:
            raise ValueError("access_token is required")
        self._access_token = access_token
        self._timeout_seconds = timeout_seconds
        self._client = client

    def run(self) -> UpstoxReadinessSnapshot:
        checks: list[ReadinessCheck] = []

        profile = self._get("/v2/user/profile")
        profile_data = profile.get("data", {}) if isinstance(profile, dict) else {}
        exchanges = tuple(str(value) for value in profile_data.get("exchanges", []) or [])
        products = tuple(str(value) for value in profile_data.get("products", []) or [])

        profile_ok = profile.get("status") == "success"
        checks.append(
            ReadinessCheck(
                name="profile_api",
                passed=profile_ok,
                detail="Upstox profile API authenticated successfully" if profile_ok else "profile API did not return success",
            )
        )
        checks.append(
            ReadinessCheck(
                name="nse_enabled",
                passed="NSE" in exchanges,
                detail="NSE is enabled on the account" if "NSE" in exchanges else "NSE is not listed in enabled exchanges",
            )
        )
        checks.append(
            ReadinessCheck(
                name="intraday_product_enabled",
                passed="I" in products,
                detail="intraday product I is enabled" if "I" in products else "intraday product I is not enabled",
            )
        )

        funds = self._get(
            "/v3/user/get-funds-and-margin",
            extra_headers={"Api-Version": "3.0"},
        )
        available_to_trade = self._extract_available_to_trade(funds)
        checks.append(
            ReadinessCheck(
                name="funds_api",
                passed=funds.get("status") == "success" and available_to_trade is not None,
                detail=(
                    f"available-to-trade balance reported by Upstox: ₹{available_to_trade}"
                    if available_to_trade is not None
                    else "funds API did not expose an available-to-trade total"
                ),
            )
        )
        checks.append(
            ReadinessCheck(
                name="minimum_test_capital_present",
                passed=available_to_trade is not None and available_to_trade >= Decimal("1000"),
                detail=(
                    "at least ₹1,000 is available to trade"
                    if available_to_trade is not None and available_to_trade >= Decimal("1000")
                    else "less than ₹1,000 is currently available to trade"
                ),
            )
        )

        static_ips = self._get("/v2/user/ip")
        ip_data = static_ips.get("data", {}) if isinstance(static_ips, dict) else {}
        primary = _first_nonempty(ip_data, "primary_ip", "primaryIp")
        secondary = _first_nonempty(ip_data, "secondary_ip", "secondaryIp")
        primary_configured = bool(primary)
        secondary_configured = bool(secondary)

        checks.append(
            ReadinessCheck(
                name="primary_static_ip",
                passed=static_ips.get("status") == "success" and primary_configured,
                detail=(
                    "primary static IP is registered with Upstox"
                    if primary_configured
                    else "primary static IP is not yet registered; live API orders remain blocked"
                ),
            )
        )

        return UpstoxReadinessSnapshot(
            checks=tuple(checks),
            available_to_trade=available_to_trade,
            exchanges=exchanges,
            products=products,
            primary_static_ip_configured=primary_configured,
            secondary_static_ip_configured=secondary_configured,
        )

    def _get(self, path: str, *, extra_headers: dict[str, str] | None = None) -> dict[str, Any]:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._access_token}",
        }
        if extra_headers:
            headers.update(extra_headers)

        url = f"{UPSTOX_API_BASE}{path}"
        if self._client is not None:
            response = self._client.get(url, headers=headers, timeout=self._timeout_seconds)
        else:
            response = httpx.get(url, headers=headers, timeout=self._timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"unexpected non-object Upstox response for {path}")
        return payload

    @staticmethod
    def _extract_available_to_trade(payload: dict[str, Any]) -> Decimal | None:
        try:
            value = payload["data"]["available_to_trade"]["total"]
        except (KeyError, TypeError):
            return None
        try:
            return Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return None


def _first_nonempty(mapping: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if value:
            return str(value)
    return None
