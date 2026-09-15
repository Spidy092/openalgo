"""Autonomous trading bridge (Phase 3 — sandbox-only, no real money).

This package is the ONLY place allowed to connect a research eligibility
decision and a signal to an OpenAlgo order dispatch. It is built safely by
construction:

- Phase 3 dispatches EXCLUSIVELY through ``services.sandbox_service.sandbox_place_order``,
  which does not read the global analyze toggle and never reaches a broker. No
  real order can be placed from this package in Phase 3.
- Every order intent must pass a validated ``EligibilityDecision`` and the full
  deny-by-default ``evaluate_all_rails`` gate from ``equity_engine`` before dispatch.
- Full-auto is disabled by default; a decision must explicitly carry
  ``autonomy_mode=full_auto`` AND the caller must pass ``allow_full_auto=True``.
- Every attempt/decision/result is appended to an audit trail.
- Dependency direction is one-way: this package imports ``equity_engine``;
  ``equity_engine`` never imports ``services``.

The live dispatch target (Phase 4+) is intentionally NOT wired here. Switching
to real orders is a separate, explicitly-approved change.
"""

from __future__ import annotations

from services.autonomous_bridge.bridge import (
    BridgeDispatchResult,
    BridgeMode,
    DispatchTarget,
    OrderIntent,
    autonomous_dispatch,
)

__all__ = [
    "BridgeDispatchResult",
    "BridgeMode",
    "DispatchTarget",
    "OrderIntent",
    "autonomous_dispatch",
]
