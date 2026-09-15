# Larger Historical Acquisition + Strategy Discovery — Design & Agent Plan (v1 DRAFT)

Status: DRAFT for owner approval. No acquisition, no orders. This plan produces
enough real intraday history to run the existing discovery gauntlet so a
strategy candidate can actually pass, unblocking Phases 4-5 of the bridge.

Owner goal: build toward a profitable, safe, autonomous trading agent. This step
is the data + discovery foundation. It spends no real money and places no order.

---

## 0. Why this step exists (verified)

- The discovery pipeline (`equity_engine.intraday_discovery.run_discovery`) requires a
  train / validation / test split by trade date PLUS multiple walk-forward folds.
- Our current real dataset (MCX pilot) is only 3 trading days — verified too small;
  it fails at `split_train_validation_test`, not on strategy quality.
- Therefore we must acquire several months of 5-minute history before discovery
  can produce a meaningful PASS/FAIL.

## 1. Verified technical constraints (from code + Upstox docs)

- Upstox minute-history: max span 28 calendar days per request
  (`UPSTOX_MINUTE_MAX_CALENDAR_DAYS = 28`), GET-only, bounded retries on
  408/425/429/5xx, >= 0.15s pacing (`upstox_batch_history.py`).
- Request math: one symbol, N calendar days => ceil(N / 28) requests.
  - 6 months (~182 days) => 7 requests/symbol.
  - 12 months (~365 days) => 13 requests/symbol.
- Rate limits are per-API/per-user bands; a few dozen paced GETs is well within them.
- Live token expires daily (~03:30 IST); a multi-request run must complete within
  a valid token window, and the executor already resumes verified chunks safely.
- Acquisition executor is already built and safe: immutable raw capture, SHA-256,
  manifest/state, resume, corruption fail-closed, no live-order path.

## 2. Scope (owner to confirm the two bracketed choices)

- Instruments: start with a SMALL set of liquid NSE cash-equity symbols.
  Recommended start: 3-5 large-caps (e.g. RELIANCE, INFY, HDFCBANK, TCS, ICICIBANK).
  [OWNER: confirm the symbol list, or accept this default.]
- Interval: 5-minute only (matches the bridge + discovery).
- History length: [OWNER: choose 6 or 12 months]. Default recommendation: 12 months
  for robust walk-forward folds; 6 months is the faster minimum.
- Window boundary: respect the existing point-in-time rules already in the engine
  (no back-projection across the 2026-08-03 CAS regime inside a single normal-session
  batch; the code enforces this).
- Track: intraday cash equity only. Swing stays blocked until Phase 6 delivery costs.

## 3. Safety and provenance (unchanged, enforced by existing code)

- Read-only market-data GET only. No order, portfolio, or funds mutation.
- Token from OpenAlgo stored auth (never printed/persisted), same as the pilot.
- Every dataset fingerprinted; raw bytes immutable; resume verified; corruption
  fails closed. `live_orders_called=false` throughout.
- New EC2 output only, under an isolated directory; existing data untouched.

## 4. Build phases (safe order)

- **Phase A (prefilter, no download):** Produce an explicit, point-in-time
  affordability/liquidity candidate file for the chosen symbols (the executor
  requires `--candidate-file` + `--prefilter-evidence`; it refuses full-universe
  execution). This is a research artifact, no network to Upstox candles.
- **Phase B (acquisition):** Run the existing `upstox_history_batch.py --execute`
  for the candidate set, 5-minute, over the chosen window, into a NEW isolated
  output dir on EC2. Token from OpenAlgo stored auth. Resume-safe.
- **Phase C (validation):** Validate each dataset with `validate_historical_dataset.py`
  (the handoff we fixed); confirm coverage, sessions, CAS, timezone, fingerprints.
- **Phase D (discovery):** Run `discover_intraday_strategy.py` per symbol with a
  proper train/validation/test split + walk-forward folds. Emit `strategy_candidate.json`
  per symbol with hard profitability gates. Most will FAIL; that is correct.
- **Phase E (eligibility):** For any candidate that PASSES all gates, build a signed
  `EligibilityDecision` (₹1000 cap, intraday, supervised_auto). This is the input
  Phase 4 would consume — still no real order.

Real money is NOT part of this plan. Phase 4 (first live ₹1000 order) remains a
separate, explicitly-approved step, and only if a candidate genuinely passes.

## 5. Honest expectation

Most simple rule templates will NOT pass out-of-sample after real costs. That is
the system working: it rejects losers. A PASS is the exception, not the default,
and even a PASS is small-capital, monitored, and loss-limited. No profit is
promised; the gates exist to avoid trading an unproven edge.

---

## 6. Three-agent work split

Kiro (me): run acquisition on EC2 (token handling, resume, isolation), validation,
integration, safety review, and the eligibility step. I keep anything that touches
the live token or EC2 execution.

The two OpenCode agents get the isolated, no-network research specs below.

---

## 7. Spec for OpenCode Agent 1 — "Prefilter + candidate file builder" (Phase A)

Paste to Agent 1.

```
GOAL: Build a repeatable, offline point-in-time prefilter that turns a small
approved symbol list into the candidate_file.json + prefilter_evidence.json the
Upstox acquisition executor requires. NO network to Upstox candles. NO orders.
Tests first.

CONSTRAINTS:
- Work only under research/equity_engine/ (src + scripts + tests).
- Do NOT modify broker/ or services/ or any live path.
- Reuse existing modules: nse_calendar, market_sessions, liquidity, universe,
  provenance. Do not reimplement them.
- Intraday cash equity only. Output records live_orders_called=false.

DELIVERABLE — scripts/build_intraday_candidate_file.py + supporting module:
- Input: an explicit list of NSE symbols + instrument keys, a start/end date
  window, and a documented liquidity/affordability rationale.
- Produce candidate_file.json (list of {instrument_key, symbol, start, end}) and
  prefilter_evidence.json (schema-tagged, affordability_prefilter_applied=true,
  with the point-in-time rationale and provenance fingerprints).
- Fail closed on: missing instrument key, window crossing the CAS regime boundary
  inside one normal-session batch, or any unresolved eligibility.

TESTS: deterministic tests proving the artifacts are well-formed and that a
bad/ambiguous input fails closed.

FORBIDDEN: any Upstox candle GET, any broker POST, any real-money action.
```

## 8. Spec for OpenCode Agent 2 — "Discovery batch runner + report" (Phase D/E prep)

Paste to Agent 2.

```
GOAL: Build a batch discovery runner that, given a directory of validated
per-symbol Parquet+manifest datasets, runs the existing discovery gauntlet on
each with a proper train/validation/test split and walk-forward folds, and
writes a consolidated discovery report plus per-symbol strategy_candidate.json.
NO network. NO orders. Tests first.

CONSTRAINTS:
- Work only under research/equity_engine/ (src + scripts + tests).
- Do NOT modify broker/ or services/ or any live path.
- Reuse run_discovery from intraday_discovery and the existing gauntlet. Do not
  reimplement gates or metrics.
- Intraday cash equity only. Output records live_orders_called=false.

DELIVERABLE — scripts/discover_intraday_batch.py + supporting module:
- Input: a directory containing one subdir per symbol, each with the dataset
  parquet + manifest produced by the acquisition executor, plus split dates.
- For each symbol: load, run run_discovery with sensible default rule templates,
  collect PASS/FAIL, net expectancy, profit factor, Sharpe, max drawdown, and
  per-fold walk-forward results.
- Emit discovery_report.json (ranked summary across symbols) and per-symbol
  strategy_candidate.json. Rank only PASS candidates; never fabricate a pass.

TESTS: deterministic tests on small synthetic multi-day fixtures proving the
runner PASSES a known-good synthetic edge and FAILS a no-edge symbol, and that
the split/walk-forward requirement is enforced (too-small data fails closed).

FORBIDDEN: any live order path, any broker POST, any real-money action.
```

---

## 9. What Kiro does after Agents 1 & 2 deliver

- Review both PRs against this plan and the safety boundary (services never
  imported by equity_engine; no live-order path).
- Run acquisition on EC2 for the approved symbols/window using the OpenAlgo stored
  token, into a new isolated dir, resume-safe, with a fresh daily token as needed.
- Validate every dataset; run the batch discovery; present the ranked report.
- If a candidate passes, build its signed EligibilityDecision and bring the Phase 4
  go/no-go to the owner. No real order without explicit approval.

## 10. Owner approvals required

1. Approve this plan (and the symbol list + 6/12-month choice).
2. Approve running the real acquisition on EC2 (read-only market data GET).
3. Later, separately: approve any Phase 4 first real ₹1000 order.
