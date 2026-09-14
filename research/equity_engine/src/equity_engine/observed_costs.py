from __future__ import annotations

from datetime import date
from decimal import Decimal, ROUND_HALF_UP

from .documented_costs import CurrentTermsNSEIntradayCostProvider
from .models import CostSource
from .upstox_costs import UPSTOX_BROKERAGE_URL


PAISA = Decimal("0.01")


class ObservedUpstoxNSEIntradayCostProvider(CurrentTermsNSEIntradayCostProvider):
    """Effective account snapshot inferred from authenticated Upstox quotes.

    This is intentionally separate from the public/documented 0.1% model. It is a
    non-authoritative research estimate whose totals must continue to be checked against
    ``UpstoxBrokerCostProvider`` before use in any research conclusion.
    """

    MODEL_NAME = "broker-observed"
    BROKERAGE_RATE = Decimal("0.0006")
    DEFAULT_EFFECTIVE_DATE = date(2026, 9, 9)
    DEFAULT_OBSERVED_AT = date(2026, 9, 9)
    SCOPE = "NSE equity intraday / this authenticated account snapshot"

    def __init__(
        self,
        *,
        pricing_date: date,
        effective_date: date = DEFAULT_EFFECTIVE_DATE,
        observed_at: date = DEFAULT_OBSERVED_AT,
    ) -> None:
        if pricing_date < effective_date:
            raise ValueError(
                "the broker-observed snapshot is effective only on or after "
                f"{effective_date.isoformat()}"
            )
        self._pricing_date = pricing_date
        self._observed_effective_date = effective_date
        self._observed_at = observed_at

    @property
    def provenance(self) -> dict[str, str]:
        return {
            "source": "Upstox authenticated Brokerage Details API",
            "observed_at": self._observed_at.isoformat(),
            "effective_date": self._observed_effective_date.isoformat(),
            "brokerage_rate": str(self.BROKERAGE_RATE),
            "scope": self.SCOPE,
        }

    def _round_component(self, value: Decimal) -> Decimal:
        return value.quantize(PAISA, rounding=ROUND_HALF_UP)

    def _cost_source(self) -> CostSource:
        return CostSource.OBSERVED_SNAPSHOT

    def _source_refs(self) -> tuple[str, ...]:
        provenance = self.provenance
        return (
            UPSTOX_BROKERAGE_URL,
            f"source={provenance['source']}",
            f"observed_at={provenance['observed_at']}",
            f"effective_date={provenance['effective_date']}",
            f"brokerage_rate={provenance['brokerage_rate']}",
            f"scope={provenance['scope']}",
        )

    def _effective_date(self) -> date:
        return self._observed_effective_date
