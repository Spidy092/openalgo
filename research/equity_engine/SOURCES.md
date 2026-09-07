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
| Upstox BOD instrument master | Upstox publishes complete BOD JSON and recommends `instrument_key` as stable unique identifier | Current docs | https://upstox.com/developer/api-documentation/instruments/ | VERIFIED; current paper/live reference only, not historical-universe authority |
| Upstox MIS eligibility | Upstox publishes a separate NSE MIS instrument JSON list | Current docs | https://upstox.com/developer/api-documentation/instruments/ | VERIFIED; current paper/live gate only, never projected backward into historical research |
| Upstox suspended instruments | Upstox publishes a separate suspended-instrument JSON list; those instruments are unavailable for trading on Upstox | Current docs | https://upstox.com/developer/api-documentation/instruments/ | VERIFIED; current paper/live gate only |
| Instrument CAS flag | BOD EQ records expose `cas_eligible` | Current docs | https://upstox.com/developer/api-documentation/instruments/ | VERIFIED; missing flag fails closed for current paper/live use |
| Instrument tick size field | BOD EQ records expose numeric `tick_size` described as minimum price movement | Current docs | https://upstox.com/developer/api-documentation/instruments/ | FIELD VERIFIED; JSON numeric unit/scale is not stated explicitly, so conversion scale is caller-supplied and unresolved below |
| Corporate actions | Upstox Corporate Actions API returns Dividend, Bonus, Split and Rights events by ISIN with ex/effective date and ratio/amount | Current docs | https://upstox.com/developer/api-documentation/get-corporate-actions/ | VERIFIED |
| NSE MII CM security file | NSE publishes `NSE_CM_security_ddmmyyyy.csv.gz` daily on the website | 2024-02-05 onward | https://nsearchives.nseindia.com/content/circulars/MSD60315.pdf | VERIFIED; this date is the initial exact point-in-time research boundary |
| NSE MII SSEC field | Annexure 10 field 11 uses ISO tag `CallAuctnInd`; NSE stated no other file-structure/format change in that modification | 2025-06-16 | https://nsearchives.nseindia.com/content/circulars/CMTR68176.pdf | VERIFIED |
| NSE MII archive implementation path | Working open-source NSE data implementation uses `/content/cm/NSE_CM_security_{ddmmyyyy}.csv.gz` and parses it as gzip CSV | Current implementation reference | https://github.com/Praglitch/market-data-feed/blob/main/packages/nse-data/src/nsedata/registry.py | CORROBORATED IMPLEMENTATION REFERENCE; not treated as regulatory authority |
| NSE MII ISO-tag schema cross-check | Independent parser recognizes `FinInstrmId`, `TckrSymb`, `SctySrs`, `ISIN`, `NewBrdLotQty`, `SctyTpFlg`, `BidIntrvl`, `CallAuctnInd`, `PrtdToTrad`, `SctyStsNrmlMkt`, `ElgbltyNrmlMkt` and schema changes around CAS | Current implementation reference | https://github.com/KamalSuman/india-swing-trading-system/blob/main/src/india_swing/reference_data/security_master.py | CORROBORATED IMPLEMENTATION REFERENCE; our parser still fails closed on missing tags/unknown semantics |
| NSE security-file tick authority | NSE states the applicable trading tick size is available in the daily security files | Current applicable rule | https://nsearchives.nseindia.com/content/circulars/CMTR67133.pdf | VERIFIED |
| NSE CM tick-size tiers | Below ₹250: ₹0.01; ₹250-₹1,000: ₹0.05; >₹1,000-₹5,000: ₹0.10; >₹5,000-₹10,000: ₹0.50; >₹10,000-₹20,000: ₹1.00; >₹20,000: ₹5.00 | 2025-04-15 onward, monthly review | https://nsearchives.nseindia.com/content/circulars/CMTR67133.pdf | VERIFIED |
| NSE cash transaction charge | ₹306.99 per crore each side | 2026-03-01 | https://nsearchives.nseindia.com/content/circulars/FA73061.pdf | VERIFIED |
| NSE cash IPFT contribution | ₹0.01 per crore each side | 2026-03-01 | https://nsearchives.nseindia.com/content/circulars/FA73061.pdf | VERIFIED |
| NSE cash combined exchange outflow | ₹307.00 per crore each side | 2026-03-01 | https://nsearchives.nseindia.com/content/circulars/FA73061.pdf | VERIFIED; model keeps transaction and IPFT separate |
| Intraday STT | 0.025% on sell side for non-delivery equity sale | Current | https://www.nseindia.com/static/invest/first-time-investor-sebi-turnover-fees-stt-other-levies | VERIFIED |
| NSE CAS applicability | Phase 1: cash stocks on which derivative contracts are available | 2026-08-03 | https://www.nseindia.com/static/products-services/closing-auction-session | VERIFIED |
| NSE CAS timing | CAS securities leave continuous session at 15:15; CAS 15:15-15:35. Non-CAS cash securities continue to 15:30 | 2026-08-03 | https://www.nseindia.com/static/products-services/closing-auction-session | VERIFIED |
| Trade-to-Trade surveillance | NSE reviews movement into Trade-to-Trade fortnightly and movement in/out quarterly | 2026-2027 calendar | https://www.nseindia.com/static/regulations/exchange-market-surveillance-actions-tt-schedule | VERIFIED; historical research needs point-in-time exchange evidence, current Upstox status is only a paper/live gate |
| VectorBT anti-lookahead | Signals generated using close should be shifted forward; execute at a price after the signal | Current docs | https://vectorbt.dev/api/portfolio/base/ | VERIFIED |
| VectorBT version | 1.1.0 | Released 2026-07-05 | https://pypi.org/project/vectorbt/ | VERIFIED |
| VectorBT dependency floor | Actual install metadata requires NumPy >=2.4.6 and pandas >=3.0.3,<4 | VectorBT 1.1.0 | GitHub Actions resolver + package metadata | VERIFIED by CI |

## Intentionally unresolved / not assumed

### NSE MII raw-code semantics and `BidIntrvl` scale

The strict `NseMiiSecurityMasterParser` stores normal-market status, normal-market eligibility, permitted-to-trade, security-type and `BidIntrvl` as raw values. It does **not** silently turn those values into historical tradability or rupee tick size.

Two independent open-source implementations corroborate the ISO tag names and one treats `BidIntrvl` as integer paise, but a third-party implementation is not enough authority for a money-critical conversion. Therefore:

- `NseMiiEligibilitySemantics` has no defaults and requires explicit known/eligible code sets plus a source.
- `to_tick_size_point` requires `bid_interval_scale_rupees_per_raw_unit` plus a source.
- unknown status or eligibility values fail closed.
- the real historical tournament cannot be labelled exact until the effective-dated code semantics and tick scale are verified or reconciled against NSE security/tick evidence.

### Historical universe before 2024-02-05

NSE/MSD/60315 verifies daily website dissemination of the MII CM security master from February 05, 2024. We do not infer the exact 2022-2024-02-04 point-in-time universe from today's broker list. Older candle data may be used for exploratory strategy behavior only until a verified legacy daily security-master source is incorporated.

### Upstox JSON tick-size numeric scale

The Upstox JSON instrument documentation describes `tick_size` as the minimum price movement but does not explicitly state whether a JSON value such as `5.0` is represented in rupees, paise, or another numeric scale. The old deprecated CSV examples use rupee-style decimal values such as `0.05`, while current JSON examples show values such as `5.0`/`10.0`.

Therefore `build_nse_equity_master` requires `tick_size_scale_rupees_per_raw_unit` from the experiment and records it in the dataset digest. Before live eligibility, the converted value must be cross-checked against the applicable NSE security-master/tick-size evidence for that instrument/date. No implicit `/100` conversion is permitted in production research code.

### Historical-candle corporate-action adjustment

The verified Upstox V3 candle documentation does not state whether OHLC history is adjusted for splits, bonuses or rights. Upstox exposes those events separately through the Corporate Actions API. Until an adjustment/normalization method is verified, research windows containing caller-designated structural events are blocked rather than silently adjusted.

### Historical statutory and broker fee schedule for 2022-2025

The current `CurrentTermsNSEIntradayCostProvider` answers a specific question:

> Would a historical signal survive the **current verified cost schedule**?

It does **not** claim to recreate the fee schedule that actually applied on each historical trade date. A separate effective-dated historical cost ledger must be researched from primary circulars before any report is labelled `HISTORICAL_ACTUAL_COSTS`.

### Delivery / swing exact cost

Swing/delivery live eligibility is blocked until DP/demat charges and all delivery-specific components are represented and reconciled against the user's actual broker DP plan. The Brokerage Details API explicitly states the DP minimum is not included in its brokerage total.

### Historical liquidity vs exact turnover

OHLCV does not contain every execution price, so the research universe uses a clearly labelled **daily notional proxy**: `sum(bar close × bar volume)`. It is useful for relative screening but must not be described as exact exchange turnover.

### Spread and slippage

No universal spread/slippage number is hard-coded. Historical OHLC is not used as a bid/ask-spread substitute. Backtests must receive explicit friction assumptions and stress scenarios. Paper/live eligibility separately requires observed bid/ask samples; observed fill/spread distributions should replace generic assumptions once enough data exists.

### NSE CAS participation

V1 deliberately avoids placing research/live orders in CAS. The session policy exits before the applicable continuous-market end using an explicit caller-supplied buffer. CAS auction-fill modelling is a separate research problem.

### Special trading sessions

Muhurat/special-session times are not guessed. `NSEEquitySessionPolicy` accepts explicit date overrides, which must be sourced from exchange notices for the relevant backtest/live date.
