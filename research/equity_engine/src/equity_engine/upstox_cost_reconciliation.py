from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable, Iterable

from .costs import CostProvider, ReconciliationResult, reconcile_costs
from .documented_costs import CurrentTermsNSEIntradayCostProvider
from .models import Exchange, OrderSpec, Product, Side
from .sizing import max_affordable_buy_quantity
from .upstox_costs import UpstoxBrokerCostProvider


EXIT_PASS = 0
EXIT_RECONCILIATION_FAILED = 1
EXIT_CONFIGURATION_ERROR = 2
EXIT_BROKER_API_ERROR = 3

DEFAULT_NOTIONALS = ("100", "250", "500", "750", "950")


@dataclass(frozen=True)
class ReconciliationItem:
    side: str
    quantity: int
    price: Decimal
    notional: Decimal
    local_modeled_total: Decimal | None
    broker_reported_total: Decimal | None
    absolute_difference: Decimal | None
    tolerance: Decimal
    status: str
    broker_charges: dict[str, Decimal] | None = None
    error: str | None = None


@dataclass(frozen=True)
class ReconciliationReport:
    schema_version: str
    generated_at_utc: str
    instrument_token: str
    symbol: str
    pricing_date: date
    capital: Decimal
    tolerance: Decimal
    items: tuple[ReconciliationItem, ...]
    overall: str
    live_orders_called: bool = False

    @property
    def passed_count(self) -> int:
        return sum(item.status == "PASS" for item in self.items)

    @property
    def failed_count(self) -> int:
        return sum(item.status == "FAIL" for item in self.items)

    @property
    def error_count(self) -> int:
        return sum(item.status == "ERROR" for item in self.items)


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def report_as_dict(report: ReconciliationReport) -> dict[str, object]:
    """Return a credential-free, JSON-safe evidence/report representation."""

    result = asdict(report)
    result["pricing_date"] = report.pricing_date.isoformat()
    result["capital"] = _decimal_text(report.capital)
    result["tolerance"] = _decimal_text(report.tolerance)
    result["items"] = [
        {
            "side": item.side,
            "quantity": item.quantity,
            "price": _decimal_text(item.price),
            "notional": _decimal_text(item.notional),
            "local_modeled_total": (
                _decimal_text(item.local_modeled_total)
                if item.local_modeled_total is not None
                else None
            ),
            "broker_reported_total": (
                _decimal_text(item.broker_reported_total)
                if item.broker_reported_total is not None
                else None
            ),
            "absolute_difference": (
                _decimal_text(item.absolute_difference)
                if item.absolute_difference is not None
                else None
            ),
            "tolerance": _decimal_text(item.tolerance),
            "status": item.status,
            "broker_charges": (
                {key: _decimal_text(value) for key, value in item.broker_charges.items()}
                if item.broker_charges is not None
                else None
            ),
            "error": item.error,
        }
        for item in report.items
    ]
    result["generated_at_utc"] = report.generated_at_utc
    result["live_orders_called"] = False
    result["summary"] = {
        "orders_checked": len(report.items),
        "passed": report.passed_count,
        "failed": report.failed_count,
        "errors": report.error_count,
    }
    return result


def report_as_json(report: ReconciliationReport) -> str:
    return json.dumps(report_as_dict(report), indent=2, sort_keys=True) + "\n"


def _safe_error(exc: Exception, *, access_token: str) -> str:
    # Provider errors are intentionally reduced to a type/message and any accidental token
    # echo is redacted before the result can reach the terminal or an evidence artifact.
    message = str(exc).replace(access_token, "[REDACTED]")
    return f"{type(exc).__name__}: {message}"


def _unique_positive_decimals(values: Iterable[Decimal]) -> tuple[Decimal, ...]:
    unique = {value for value in values}
    return tuple(sorted(unique))


def build_orders(
    *,
    instrument_token: str,
    price: Decimal,
    capital: Decimal,
    target_notionals: Iterable[Decimal],
    cost_provider: CostProvider,
) -> tuple[OrderSpec, ...]:
    if not instrument_token.strip():
        raise ValueError("--instrument-token is required")
    if not price.is_finite() or price <= 0:
        raise ValueError("--price must be a positive finite Decimal")
    if not capital.is_finite() or capital <= 0:
        raise ValueError("--capital must be a positive finite Decimal")

    orders: list[OrderSpec] = []
    seen_quantities: set[int] = set()
    for target in _unique_positive_decimals(target_notionals):
        if not target.is_finite() or target <= 0:
            raise ValueError("target notionals must be positive finite Decimals")
        affordability = max_affordable_buy_quantity(
            instrument_token=instrument_token.strip(),
            exchange=Exchange.NSE,
            product=Product.INTRADAY,
            price=price,
            cash_limit=min(target, capital),
            cost_provider=cost_provider,
        )
        quantity = affordability.quantity
        if quantity <= 0 or quantity in seen_quantities:
            continue
        seen_quantities.add(quantity)
        for side in (Side.BUY, Side.SELL):
            orders.append(
                OrderSpec(
                    instrument_token=instrument_token.strip(),
                    exchange=Exchange.NSE,
                    side=side,
                    product=Product.INTRADAY,
                    quantity=quantity,
                    price=price,
                )
            )

    if not orders:
        raise ValueError("no affordable positive quantities were generated from the supplied price")
    return tuple(orders)


def reconcile_orders(
    *,
    access_token: str,
    instrument_token: str,
    symbol: str | None,
    price: Decimal,
    capital: Decimal,
    pricing_date: date,
    tolerance: Decimal,
    target_notionals: Iterable[Decimal],
    local_provider: CostProvider | None = None,
    broker_provider: CostProvider | None = None,
    now: Callable[[], datetime] | None = None,
) -> ReconciliationReport:
    if not access_token.strip():
        raise ValueError("UPSTOX_ACCESS_TOKEN is not set")
    if not tolerance.is_finite() or tolerance < 0:
        raise ValueError("--tolerance must be a non-negative finite Decimal")

    local = local_provider or CurrentTermsNSEIntradayCostProvider(pricing_date=pricing_date)
    orders = build_orders(
        instrument_token=instrument_token,
        price=price,
        capital=capital,
        target_notionals=target_notionals,
        cost_provider=local,
    )
    broker = broker_provider or UpstoxBrokerCostProvider(access_token=access_token)
    generated_at = (now or (lambda: datetime.now(timezone.utc)))().astimezone(timezone.utc)

    items: list[ReconciliationItem] = []
    for order in orders:
        local_quote = local.quote(order)
        try:
            broker_quote = broker.quote(order)
            if broker_quote.broker_reported_total is None:
                raise ValueError("broker quote did not include charges.total")
            reconciliation: ReconciliationResult = reconcile_costs(
                local_quote,
                broker_quote,
                tolerance=tolerance,
            )
        except Exception as exc:  # one failed API order must not become a false PASS
            items.append(
                ReconciliationItem(
                    side=order.side.value,
                    quantity=order.quantity,
                    price=order.price,
                    notional=order.notional,
                    local_modeled_total=local_quote.total,
                    broker_reported_total=None,
                    absolute_difference=None,
                    tolerance=tolerance,
                    status="ERROR",
                    error=_safe_error(exc, access_token=access_token),
                )
            )
            continue

        items.append(
            ReconciliationItem(
                side=order.side.value,
                quantity=order.quantity,
                price=order.price,
                notional=order.notional,
                local_modeled_total=reconciliation.estimated_total,
                broker_reported_total=reconciliation.broker_total,
                absolute_difference=reconciliation.absolute_error,
                tolerance=reconciliation.tolerance,
                status="PASS" if reconciliation.passed else "FAIL",
                broker_charges={name: value for name, value in vars(broker_quote.charges).items()},
            )
        )

    if any(item.status == "ERROR" for item in items):
        overall = "ERROR"
    elif any(item.status == "FAIL" for item in items):
        overall = "FAIL"
    else:
        overall = "PASS"

    return ReconciliationReport(
        schema_version="upstox-cost-reconciliation/v1",
        generated_at_utc=generated_at.isoformat(),
        instrument_token=instrument_token.strip(),
        symbol=symbol.strip() if symbol and symbol.strip() else instrument_token.strip(),
        pricing_date=pricing_date,
        capital=capital,
        tolerance=tolerance,
        items=tuple(items),
        overall=overall,
        live_orders_called=False,
    )


def exit_code_for(report: ReconciliationReport) -> int:
    if report.overall == "PASS":
        return EXIT_PASS
    if report.overall == "FAIL":
        return EXIT_RECONCILIATION_FAILED
    return EXIT_BROKER_API_ERROR


def render_terminal(report: ReconciliationReport) -> str:
    lines = [
        "UPSTOX COST RECONCILIATION",
        "",
        f"Instrument: {report.symbol} / {report.instrument_token}",
        f"Pricing date: {report.pricing_date.isoformat()}",
        f"Capital cap: ₹{_decimal_text(report.capital)}",
        f"Tolerance: ₹{_decimal_text(report.tolerance)} (configured)",
        "",
    ]
    current_side: str | None = None
    for item in report.items:
        if item.side != current_side:
            current_side = item.side
            lines.append(current_side)
        lines.extend(
            [
                f"qty={item.quantity} price={_decimal_text(item.price)}",
                f"notional={_decimal_text(item.notional)}",
                f"local={_decimal_text(item.local_modeled_total) if item.local_modeled_total is not None else 'n/a'}",
                f"broker={_decimal_text(item.broker_reported_total) if item.broker_reported_total is not None else 'n/a'}",
                f"difference={_decimal_text(item.absolute_difference) if item.absolute_difference is not None else 'n/a'}",
                f"tolerance={_decimal_text(item.tolerance)}",
                item.status,
            ]
        )
        if item.error:
            lines.append(f"error={item.error}")
        lines.append("")
    lines.extend(
        [
            "-------------------------",
            f"Orders checked: {len(report.items)}",
            f"Passed: {report.passed_count}",
            f"Failed: {report.failed_count}",
            f"Errors: {report.error_count}",
            f"Overall: {report.overall}",
            "Live orders called: false",
        ]
    )
    return "\n".join(lines) + "\n"


def write_evidence(path: Path, report: ReconciliationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_as_json(report), encoding="utf-8")
