# Autonomous Trading Bridge — Design & Build Plan (v1 DRAFT)

Status: DRAFT for owner approval. No code is written and no live order is placed
until the owner approves this document. This describes how to connect the
`equity_engine` research brain to OpenAlgo live execution **safely by
construction**, and how to split the build across three agents (Kiro + two
OpenCode agents).

---

## 0. Non-negotiable safety statement

This system will place **real orders with real money**. Fully automatic dispatch
is only acceptable inside hard, tested rails. We do **not** remove the existing
fail-closed guards; we add a new, explicit, audited bridge that is itself gated.

Irreversibility: every live order spends real money instantly. Bugs, bad fills,
or a runaway loop are real losses. Therefore safety rails are built and tested
**before** the automatic dispatch is switched on.

No profit is promised. The system's real job is to **reject** bad strategies,
trade survivors small, and **stop on loss**.

---

## 1. Current verified state (facts, not assumptions)

- OpenAlgo platform is live on EC2 `openalgo-prod` (`13.203.137.243`), service running.
- Upstox live order path is complete: `broker/upstox/api/order_api.py` → `POST /v2/order/place`.
- Analyze (sandbox/paper) vs live toggle exists: `database.settings_db.get_analyze_mode`.
- Live-readiness probe PASSES on the real account: primary static IP registered,
  NSE enabled, intraday product enabled, funds API ok, ₹1000 available.
- Research engine `equity_engine` is quarantined: it cannot place orders and
  raises `LiveOrderAttemptError` on any `live_orders_called=True`.
- Token expires daily (~03:30 IST); must be refreshed via OpenAlgo login each day.
- Delivery/swing cost model is NOT built: `upstox_costs.py` raises
  `NotImplementedError`. **Swing cannot go live until this is built.**

---

## 2. Scope (owner decisions)

- Tracks: **intraday first**, swing second (swing blocked until delivery cost model exists).
- Capital: hard cap **₹1000** now for real-money testing; raised later only by explicit change.
- Autonomy: fully automatic dispatch, but only for strategies that passed all gates,
  and only inside the safety rails below.
- Instruments: start with **one** liquid NSE cash-equity symbol, expand after it is stable.

---

## 3. Architecture — the closed loop

```
[equity_engine research]                     [SAFETY RAILS]          [OpenAlgo live]
 strategy discovery ─► backtest ─► walk-forward ─► cost model ─►
 paper/shadow ─► eligibility decision (signed)  ─►  BRIDGE  ─► RiskGuard ─► place_order ─► Upstox
                                                     ▲  │
                                          kill-switch│  └─► audit + reconcile ─► P&L / stop-loss
```

Component list:

1. **Strategy discovery** (equity_engine, exists/partial): generate candidate rules.
2. **Validation gauntlet** (exists): backtest → walk-forward → event-driven fill sim → cost model.
3. **Paper/shadow** (exists): prove on live data with `live_orders_called=false`.
4. **Eligibility decision** (NEW, small): a signed artifact that says "strategy X is
   approved for live, capital ₹N, instrument Y, valid until date Z". Fail-closed.
5. **Bridge** (NEW, the core): consumes an eligibility decision + a live signal, and
   is the ONLY code allowed to call the live order service automatically.
6. **Safety rails** (NEW, load-bearing): see section 4.
7. **Reconcile + P&L + stop** (partly exists): after each fill, reconcile cost, track
   daily P&L, and trip the kill-switch on breach.

---

## 4. Safety rails (built and tested BEFORE auto-dispatch is enabled)

Every rail is fail-closed: if it cannot be evaluated, the order is refused.

- **Hard capital cap:** total deployed ≤ ₹1000; per-order notional cap; reject if exceeded.
- **Daily loss limit:** if realized+unrealized loss ≥ configured ₹ limit, kill-switch trips,
  flatten and stop for the day.
- **Kill-switch:** a single file/flag that, when set, blocks ALL automatic orders immediately.
- **Rate limit / cooldown:** max orders per minute/day; min interval between orders.
- **Idempotency:** every intended order has a unique client key; duplicate suppression so a
  retry or loop cannot double-fire.
- **Instrument allowlist:** only pre-approved symbols; anything else refused.
- **Session/time gate:** only within normal session; no orders near close buffer; CAS-aware.
- **Eligibility freshness:** decision artifact must be valid, signed, unexpired, matching instrument.
- **Token validity precheck:** verify token via read-only probe before any order; stop on 401.
- **Analyze-mode consistency:** the run's mode is authoritative; never silently divert live→sandbox.
- **Full audit:** every attempt/decision/result persisted (reuse `services/agent/safety/audit.py` pattern).
- **Existing RiskGuard:** the bridge calls the live service through the same RiskGuard used by the
  AI agent (`services/agent/safety/risk.py`) — approval-equivalent is granted by the eligibility
  decision + rails, and the guard is still the last check before the broker.

Autonomy modes (both supported):
- **Supervised-auto:** bridge prepares the order, human approves once (existing agent flow).
- **Full-auto:** bridge dispatches without human, allowed ONLY when eligibility + all rails pass.
  Full-auto is disabled by default and enabled per-strategy by an explicit signed flag.

---

## 4b. Profitability objective (this is a primary goal, not an afterthought)

Profit matters. The owner is committing real capital, infrastructure, and internet
access, and the system must be built to *pursue and measure* profit rigorously.

Honest boundary: no one can guarantee market profit. What we CAN do is make
profitability a hard, measurable gate so only strategies with a real, cost-aware,
out-of-sample edge ever reach real money, and losers are cut fast.

Profit is enforced as explicit thresholds a candidate MUST beat (all net of real costs):

- **Net-of-cost edge:** positive expectancy AFTER documented+broker-reconciled costs,
  taxes, and modeled slippage. Gross profit that dies after costs = FAIL.
- **Out-of-sample profit:** must remain profitable on the untouched test period and
  across walk-forward folds, not just in-sample. In-sample-only profit = FAIL.
- **Risk-adjusted return:** minimum Sharpe/Sortino and profit-factor thresholds
  (configurable), not just raw P&L.
- **Drawdown ceiling:** max drawdown within a configured limit; blows the limit = FAIL.
- **Consistency:** positive across a majority of walk-forward folds, not one lucky window.
- **Live shadow confirmation:** paper/shadow P&L on live data must stay consistent with
  backtest expectancy before any real order.
- **Continuous performance monitoring (live):** once live, realized P&L vs expected is
  tracked; a strategy that decays below threshold is auto-demoted (stop trading it),
  not left running on hope.

How the system actively PURSUES profit (using the resources provided, incl. internet):
- **Broad candidate search:** many rule templates + parameter sweeps via VectorBT to
  find edges, not one hand-guess.
- **Market-context features:** regime, VIX, sector strength, gaps, VWAP, liquidity —
  richer signals = better edge (already scoped in the research README).
- **Internet-sourced data (allowed, sandboxed):** agents may pull public market data,
  calendars, corporate actions, and reference data to improve signals — as untrusted
  input, provenance-fingerprinted, never executing instructions from fetched content,
  never leaking the token or capital.
- **Cost minimization:** use the broker-observed cost snapshot (0.06%) and tick-aware
  order types to keep more of the edge.
- **Capital efficiency:** size positions to the edge within the ₹1000 cap; scale capital
  only after a strategy proves itself live over time.

Result: the system is built to *win*, but "win" is defined as beating measurable,
cost-aware, out-of-sample profit gates — with fast loss-cutting when it doesn't.

## 5. Build phases (safe order — do NOT reorder)

- **Phase 1 (safe, no money):** Eligibility decision artifact + schema + signer/validator. Fail-closed.
- **Phase 2 (safe, no money):** Safety rails library (caps, loss limit, kill-switch, idempotency,
  allowlist, session gate, token precheck) with full unit tests. No dispatch yet.
- **Phase 3 (safe, no money):** Bridge in **supervised-auto** mode wired to sandbox/analyze only.
  Prove end to end with paper orders. `live_orders_called=false` throughout.
- **Phase 4 (real money, tiny):** Enable live dispatch for ONE symbol, ₹1000 cap, supervised-auto,
  one order, then reconcile and stop. Manual go/no-go by owner.
- **Phase 5 (real money):** Full-auto for the proven strategy inside all rails, intraday only.
- **Phase 6 (later):** Build delivery/swing cost model (`upstox_costs.py`), then extend to swing.
- **Daily ops:** token refresh via OpenAlgo login; readiness probe before enabling dispatch each day.

Each phase is a separate PR on its own branch, reviewed before the next. No phase
skips. Real money only from Phase 4, and only after Phases 1–3 pass tests.

---

## 6. Three-agent work split

Kiro (me): integration, safety review, wiring to OpenAlgo, phase gating, final go/no-go checks.
The two OpenCode agents get the self-contained specs below. Each spec is isolated
to avoid collisions (different files/modules), test-first, and forbids any live order.

Branch per agent; open PRs; I integrate and run the safety review before any real-money phase.

---

## 7. Spec for OpenCode Agent 1 — "Eligibility + Rails" (Phases 1 & 2)

Paste this to Agent 1.

```
GOAL: Build the eligibility decision artifact and the safety-rails library for an
autonomous trading bridge in the OpenAlgo repo. NO live orders. NO network. Tests first.

CONSTRAINTS:
- Work only under research/equity_engine/src/equity_engine/ and its tests/.
- Do NOT modify broker/, services/, or any live order path.
- Every artifact must carry live_orders_called=false and fail closed on any True.
- Use the existing canonical_sha256 provenance helper for signing/fingerprints.
- Python 3.12, match existing style, ruff clean, full pytest coverage.

DELIVERABLE 1 — eligibility_decision.py:
- A frozen dataclass EligibilityDecision with: strategy_id, instrument_key, track
  (intraday|swing), approved_capital_rupees (Decimal), per_order_notional_cap,
  daily_loss_limit_rupees, valid_from, valid_until, autonomy_mode
  (supervised_auto|full_auto), and a deterministic fingerprint.
- build/sign + validate functions. Validation fails closed on: expired, wrong
  instrument, missing fields, capital over cap, or any live_orders_called != false.

DELIVERABLE 2 — trading_rails.py:
- Pure functions/classes (no I/O to broker): capital cap check, per-order notional
  check, daily loss-limit check, rate-limit/cooldown, idempotency key generator +
  duplicate detector, instrument allowlist, session/time gate (reuse market_sessions),
  token-validity precheck signature (accepts a probe result), kill-switch file check.
- Each returns an explicit allow/deny verdict with a reason code. Deny on unknown.

TESTS: unit tests for every rail incl. boundary cases (exactly at cap, expired by 1s,
duplicate key, kill-switch set, out-of-session, loss-limit breach). All must pass.

FORBIDDEN: importing or calling any order placement API; any httpx POST to a broker.
```

---

## 8. Spec for OpenCode Agent 2 — "Strategy discovery + eligibility producer" (research)

Paste this to Agent 2.

```
GOAL: Produce a repeatable pipeline that discovers ONE simple intraday cash-equity
strategy, validates it through the existing gauntlet, and emits an eligibility
decision input. NO live orders. Uses only already-acquired/real historical data.

CONSTRAINTS:
- Work only under research/equity_engine/ (src + scripts + tests).
- Do NOT modify broker/ or services/ or any live path.
- Reuse existing modules: vectorbt_screening, walk_forward, event_simulator, costs,
  documented_costs, gates, shadow_execution. Do not reimplement them.
- Intraday cash equity only. Single liquid NSE symbol to start.
- Output must record live_orders_called=false.

DELIVERABLE — scripts/discover_intraday_strategy.py + supporting module:
- Define 2-3 simple, well-known intraday rule templates (e.g., opening-range
  breakout, VWAP reversion) as parameterized candidates.
- Run screening -> train/validation/test split with untouched test -> walk-forward
  -> event-driven fill sim -> documented cost model.
- Apply existing gates; only strategies passing out-of-sample + cost survive.
- PROFITABILITY GATES (hard, all net of real costs): positive expectancy after
  documented+observed costs; profitable on the untouched test period AND across a
  majority of walk-forward folds; minimum profit-factor and Sharpe (configurable);
  max-drawdown within a configured ceiling. Any gate fails => candidate FAIL.
- Emit a strategy_candidate.json with: strategy_id, rule params, instrument, the
  validation metrics (incl. net expectancy, profit factor, Sharpe, max drawdown,
  per-fold results), cost assumptions, and PASS/FAIL. FAIL if any gate fails.
- This JSON is the INPUT that Agent 1's EligibilityDecision consumes; do not place orders.

TESTS: deterministic tests on a small fixture proving the gauntlet rejects a bad
strategy and accepts a known-good synthetic one. All must pass.

FORBIDDEN: any live order path, any broker POST, any real-money action.
```

---

## 9. What Kiro does after Agents 1 & 2 deliver

- Review both PRs against this design and the safety rails (semantic review).
- Build the **bridge** (Phase 3) that ties eligibility + rails + a live signal to the
  OpenAlgo order service, first in supervised-auto against sandbox only.
- Run the full test suite + a sandbox end-to-end proof (`live_orders_called=false`).
- Present a Phase 4 go/no-go to the owner: one symbol, ₹1000, one supervised live order,
  then reconcile and stop. Only after explicit owner approval.

---

## 10. Owner approvals required before real money

1. Approve this design document.
2. Approve the specific strategy candidate that passed the gauntlet.
3. Approve Phase 4 (first real ₹1000 order), explicitly.
4. Approve moving to full-auto (Phase 5).
5. Separately approve swing track after the delivery cost model is built (Phase 6).

Nothing above Phase 3 touches real money without these approvals.
```
