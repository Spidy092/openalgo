from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

from .models import (
    ChargeBreakdown,
    CostQuote,
    CostSource,
    Exchange,
    OrderSpec,
    Product,
    Side,
)


UPSTOX_PRICING_SOURCE = "https://upstox.com/brokerage-charges/"
NSE_TRANSACTION_SOURCE = "https://nsearchives.nseindia.com/content/circulars/FA73061.pdf"
NSE_STT_SOURCE = (
    "https://www.nseindia.com/static/invest/first-time-investor-sebi-turnover-fees-stt-other-levies"
)

_CRORE = Decimal("10000000")


class CurrentTermsNSEIntradayCostProvider:
    """Documented current-terms estimate for NSE cash-equity intraday orders.

    This provider intentionally applies one effective-dated pricing snapshot to any supplied
    simulated price. It is useful for asking, "Would this historical trade survive today's
    costs?" It does NOT claim to reproduce the historical fee schedule for the candle date.

    The broker's Brokerage Details API remains the authoritative source for live eligibility.
    No per-component paisa/rupee rounding is guessed here; Decimal precision is preserved and
    the resulting estimate must later be reconciled against broker quotes.
    """

    EFFECTIVE_FROM = date(2026, 3, 1)

    BROKERAGE_RATE = Decimal("0.001")  # 0.1% per executed intraday order
    BROKERAGE_CAP = Decimal("20")
    STT_SELL_RATE = Decimal("0.00025")  # 0.025%, sell side only
    STAMP_BUY_RATE = Decimal("0.00003")  # 0.003%, buy side only
    NSE_TRANSACTION_RATE = Decimal("306.99") / _CRORE
    NSE_IPFT_RATE = Decimal("0.01") / _CRORE
    SEBI_TURNOVER_RATE = Decimal("10") / _CRORE
    GST_RATE = Decimal("0.18")

    def __init__(self, *, pricing_date: date) -> None:
        if pricing_date < self.EFFECTIVE_FROM:
            raise ValueError(
                "this provider contains only the cost schedule effective from 2026-03-01; "
                "use an explicit historical schedule for earlier pricing dates"
            )
        self._pricing_date = pricing_date

    def quote(self, order: OrderSpec) -> CostQuote:
        if order.exchange is not Exchange.NSE:
            raise NotImplementedError("current documented snapshot supports NSE only")
        if order.product is not Product.INTRADAY:
            raise NotImplementedError("current documented snapshot supports intraday only")

        turnover = order.notional
        brokerage = min(turnover * self.BROKERAGE_RATE, self.BROKERAGE_CAP)
        transaction = turnover * self.NSE_TRANSACTION_RATE
        ipft = turnover * self.NSE_IPFT_RATE
        sebi_turnover = turnover * self.SEBI_TURNOVER_RATE
        stt = turnover * self.STT_SELL_RATE if order.side is Side.SELL else Decimal("0")
        stamp_duty = (
            turnover * self.STAMP_BUY_RATE if order.side is Side.BUY else Decimal("0")
        )

        # Upstox's current detailed pricing table states GST for equity intraday is levied on
        # brokerage + transaction charges + IPFT. This exact formula must still be reconciled
        # against the broker quote before any live promotion.
        gst = (brokerage + transaction + ipft) * self.GST_RATE

        return CostQuote(
            order=order,
            charges=ChargeBreakdown(
                brokerage=brokerage,
                gst=gst,
                stt=stt,
                stamp_duty=stamp_duty,
                transaction=transaction,
                ipft=ipft,
                sebi_turnover=sebi_turnover,
            ),
            source=CostSource.DOCUMENTED_SNAPSHOT,
            retrieved_at=datetime.now(timezone.utc),
            source_refs=(UPSTOX_PRICING_SOURCE, NSE_TRANSACTION_SOURCE, NSE_STT_SOURCE),
            effective_date=self.EFFECTIVE_FROM,
        )
