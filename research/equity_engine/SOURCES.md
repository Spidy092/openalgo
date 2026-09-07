# Research Evidence Ledger

This ledger records external facts that affect money, execution, market structure, or reproducibility. Code must not silently replace an unresolved item with a guess.

Verified on: **2026-09-07 (Asia/Kolkata)**

| Area | Fact used by research engine | Effective / availability | Primary source | Status |
|---|---|---|---|---|
| Upstox historical candles | V3 minute candles available from Jan 2022; 1-15 minute requests capped at one month | Current docs | https://upstox.com/developer/api-documentation/v3/get-historical-candle-data/ | VERIFIED |
| Candle timestamp semantics | `data.candle[0]` is the **start time** of the candle timeframe | Current docs | https://upstox.com/developer/api-documentation/v3/get-historical-candle-data/ | VERIFIED |
| Broker cost authority | `/v2/charges/brokerage` returns total plus brokerage/tax/other-charge breakdown | Current docs | https://upstox.com/developer/api-documentation/get-brokerage/ | VERIFIED |
| Delivery DP cost caveat | `dp_plan.min_expense` is a daily minimum per scrip on sales and is **not included** in brokerage calculations | Current docs | https://upstox.com/developer/api-documentation/get-brokerage/ | VERIFIED; delivery live-cost provider intentionally blocked |
| Upstox equity intraday brokerage | ₹20 or 0.1% per executed order, whichever is lower | Current pricing | https://upstox.com/brokerage-charges/ | VERIFIED, still reconcile with broker quote |
| NSE cash transaction charge | ₹306.99 per crore each side | 2026-03-01 | https://nsearchives.nseindia.com/content/circulars/FA73061.pdf | VERIFIED |
| NSE cash IPFT contribution | ₹0.01 per crore each side | 2026-03-01 | https://nsearchives.nseindia.com/content/circulars/FA73061.pdf | VERIFIED |
| NSE cash combined exchange outflow | ₹307.00 per crore each side | 2026-03-01 | https://nsearchives.nseindia.com/content/circulars/FA73061.pdf | VERIFIED; model keeps transaction and IPFT separate |
| Intraday STT | 0.025% on sell side for non-delivery equity sale | Current | https://www.nseindia.com/static/invest/first-time-investor-sebi-turnover-fees-stt-other-levies | VERIFIED |
| NSE CAS applicability | Phase 1: cash stocks on which derivative contracts are available | 2026-08-03 | https://www.nseindia.com/static/products-services/closing-auction-session | VERIFIED |
| NSE CAS timing | CAS securities leave continuous session at 15:15; CAS 15:15-15:35. Non-CAS cash securities continue to 15:30 | 2026-08-03 | https://www.nseindia.com/static/products-services/closing-auction-session | VERIFIED |
| VectorBT anti-lookahead | Signals generated using close should be shifted forward; execute at a price after the signal | Current docs | https://vectorbt.dev/api/portfolio/base/ | VERIFIED |
| VectorBT version | 1.1.0 | Released 2026-07-05 | https://pypi.org/project/vectorbt/ | VERIFIED |
| VectorBT dependency floor | Actual install metadata requires NumPy >=2.4.6 and pandas >=3.0.3,<4 | VectorBT 1.1.0 | GitHub Actions resolver + package metadata | VERIFIED by CI |

## Intentionally unresolved / not assumed

### Historical statutory and broker fee schedule for 2022-2025

The current `CurrentTermsNSEIntradayCostProvider` answers a specific question:

> Would a historical signal survive the **current verified cost schedule**?

It does **not** claim to recreate the fee schedule that actually applied on each historical trade date. A separate effective-dated historical cost ledger must be researched from primary circulars before any report is labelled `HISTORICAL_ACTUAL_COSTS`.

### Delivery / swing exact cost

Swing/delivery live eligibility is blocked until DP/demat charges and all delivery-specific components are represented and reconciled against the user's actual broker DP plan. The Brokerage Details API explicitly states the DP minimum is not included in its brokerage total.

### Spread and slippage

No universal spread/slippage number is hard-coded. Backtests must receive explicit assumptions and run stress scenarios. Once live/paper market depth data is available, observed fill/spread distributions should replace generic assumptions.

### NSE CAS participation

V1 deliberately avoids placing research/live orders in CAS. The session policy exits before the applicable continuous-market end using an explicit caller-supplied buffer. CAS auction-fill modelling is a separate research problem.

### Special trading sessions

Muhurat/special-session times are not guessed. `NSEEquitySessionPolicy` accepts explicit date overrides, which must be sourced from exchange notices for the relevant backtest/live date.
