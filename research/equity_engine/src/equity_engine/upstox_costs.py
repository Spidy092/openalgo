from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Mapping

import httpx

from .models import (
    ChargeBreakdown,
    CostQuote,
    CostSource,
    OrderSpec,
    Product,
)


UPSTOX_BROKERAGE_URL = "https://api.upstox.com/v2/charges/brokerage"
UPSTOX_BROKERAGE_DOC = "https://upstox.com/developer/api-documentation/get-brokerage/"


def _money(value: object | None) -> Decimal:
    if value is None:
        return Decimal("0")
    try:
        parsed = Decimal(str(value))
    except (ArithmeticError, ValueError) as exc:
        raise ValueError("malformed numeric value in Upstox brokerage response") from exc
    if not parsed.is_finite() or parsed < 0:
        raise ValueError("malformed non-negative money value in Upstox brokerage response")
    return parsed


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"malformed {name} in Upstox brokerage response")
    return value


class UpstoxBrokerCostProvider:
    """Authoritative per-order cost quote from Upstox.

    Initial live scope is deliberately restricted to equity intraday. Upstox documents that
    the DP plan minimum expense shown in the brokerage response is not included in brokerage
    calculations. Delivery/swing trading therefore stays blocked here until a complete DP-cost
    treatment is implemented and reconciled.
    """

    def __init__(
        self,
        *,
        access_token: str,
        timeout_seconds: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not access_token:
            raise ValueError("access_token is required")
        self._access_token = access_token
        self._timeout_seconds = timeout_seconds
        self._client = client

    def quote(self, order: OrderSpec) -> CostQuote:
        if order.product is not Product.INTRADAY:
            raise NotImplementedError(
                "delivery cost quoting is intentionally blocked until DP/demat costs are fully modeled"
            )

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._access_token}",
        }
        params = {
            "instrument_token": order.instrument_token,
            "quantity": order.quantity,
            "product": order.product.value,
            "transaction_type": order.side.value,
            "price": str(order.price),
        }

        if self._client is not None:
            response = self._client.get(
                UPSTOX_BROKERAGE_URL,
                headers=headers,
                params=params,
                timeout=self._timeout_seconds,
            )
        else:
            response = httpx.get(
                UPSTOX_BROKERAGE_URL,
                headers=headers,
                params=params,
                timeout=self._timeout_seconds,
            )
        response.raise_for_status()
        payload = response.json()

        payload = _mapping(payload, name="payload")
        if payload.get("status") != "success":
            raise RuntimeError("Upstox brokerage response was not successful")

        data = _mapping(payload.get("data"), name="data")
        raw = _mapping(data.get("charges"), name="charges")
        if "total" not in raw or raw["total"] is None:
            raise ValueError("missing total in Upstox brokerage response")
        taxes_value = raw.get("taxes")
        taxes = _mapping({} if taxes_value is None else taxes_value, name="taxes")
        other_charges_value = raw.get("other_charges")
        if other_charges_value is None:
            other_charges_value = raw.get("otherTaxes")
        other_charges = _mapping(
            {} if other_charges_value is None else other_charges_value,
            name="other charges",
        )

        charges = ChargeBreakdown(
            brokerage=_money(raw.get("brokerage")),
            gst=_money(taxes.get("gst")),
            stt=_money(taxes.get("stt")),
            stamp_duty=_money(taxes.get("stamp_duty")),
            transaction=_money(other_charges.get("transaction")),
            clearing=_money(other_charges.get("clearing")),
            ipft=_money(other_charges.get("ipft")),
            sebi_turnover=_money(other_charges.get("sebi_turnover")),
        )

        return CostQuote(
            order=order,
            charges=charges,
            source=CostSource.BROKER_QUOTE,
            retrieved_at=datetime.now(timezone.utc),
            source_refs=(UPSTOX_BROKERAGE_DOC,),
            broker_reported_total=_money(raw.get("total")),
        )
