"""Evidence-first NSE equity research engine.

This package is research-only and intentionally contains no live-order execution entry point.
"""

from .costs import reconcile_costs, round_trip_result
from .gates import evaluate_promotion_gate
from .models import (
    ChargeBreakdown,
    CostQuote,
    CostSource,
    Exchange,
    ExecutionFriction,
    OrderSpec,
    Product,
    Side,
)

__all__ = [
    "ChargeBreakdown",
    "CostQuote",
    "CostSource",
    "Exchange",
    "ExecutionFriction",
    "OrderSpec",
    "Product",
    "Side",
    "evaluate_promotion_gate",
    "reconcile_costs",
    "round_trip_result",
]
