from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Iterable

import httpx

from .universe import CorporateActionAssessment


UPSTOX_CORPORATE_ACTIONS_DOC = (
    "https://upstox.com/developer/api-documentation/get-corporate-actions/"
)
UPSTOX_CORPORATE_ACTIONS_BASE = "https://api.upstox.com/v2/fundamentals"


@dataclass(frozen=True)
class CorporateActionEvent:
    name: str
    effective_date: date
    amount: Decimal | None
    ratio: str | None


class UpstoxCorporateActionProvider:
    """Read corporate actions by ISIN; no price adjustment is inferred here."""

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

    def fetch(self, isin: str) -> tuple[CorporateActionEvent, ...]:
        if not isin:
            raise ValueError("isin is required")
        url = f"{UPSTOX_CORPORATE_ACTIONS_BASE}/{isin}/corporate-actions"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._access_token}",
        }
        if self._client is None:
            response = httpx.get(url, headers=headers, timeout=self._timeout_seconds)
        else:
            response = self._client.get(url, headers=headers, timeout=self._timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "success":
            raise RuntimeError(f"unexpected Upstox corporate-actions response: {payload!r}")
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise RuntimeError("Upstox corporate-actions response did not contain data array")
        return parse_corporate_action_rows(rows)


def _parse_effective_date(value: object) -> date:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("corporate action missing expiry_date")
    try:
        return datetime.strptime(value.strip(), "%d %b %Y").date()
    except ValueError as exc:
        raise ValueError(f"unsupported corporate-action date format: {value!r}") from exc


def parse_corporate_action_rows(rows: Iterable[object]) -> tuple[CorporateActionEvent, ...]:
    events: list[CorporateActionEvent] = []
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("corporate action row must be an object")
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("corporate action row missing name")
        amount_raw = raw.get("amount")
        amount = Decimal(str(amount_raw)) if amount_raw is not None else None
        ratio_raw = raw.get("ratio")
        if ratio_raw is not None and not isinstance(ratio_raw, str):
            raise ValueError("corporate action ratio must be text or null")
        events.append(
            CorporateActionEvent(
                name=name.strip(),
                effective_date=_parse_effective_date(raw.get("expiry_date")),
                amount=amount,
                ratio=ratio_raw,
            )
        )
    return tuple(sorted(events, key=lambda item: (item.effective_date, item.name)))


def assess_corporate_actions(
    *,
    events: Iterable[CorporateActionEvent],
    research_start: date,
    research_end: date,
    blocked_event_names: frozenset[str],
) -> CorporateActionAssessment:
    """Mark structural events as blockers using an explicit caller-supplied policy.

    Upstox documents `expiry_date` as the ex/effective date. This function does not assume which
    event types need adjustment: `blocked_event_names` is mandatory experiment policy. A typical
    research policy may block Split/Bonus/Rights until a normalization method is verified.
    """

    if research_start > research_end:
        raise ValueError("research_start must be on or before research_end")
    if not blocked_event_names:
        raise ValueError("blocked_event_names must be explicitly non-empty")

    blocking: list[str] = []
    for event in events:
        if research_start <= event.effective_date <= research_end and event.name in blocked_event_names:
            detail = f"{event.name}@{event.effective_date.isoformat()}"
            if event.ratio:
                detail += f" ratio={event.ratio}"
            blocking.append(detail)

    return CorporateActionAssessment(complete=True, blocking_events=tuple(blocking))
