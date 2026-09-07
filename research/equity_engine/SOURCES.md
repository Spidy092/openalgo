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
| Instrument tick size field | BOD EQ records expose numeric `tick_size` described as minimum price movement | Current docs | https://upstox.com/developer/api-documentation/instruments/ | FIELD VERIFIED; current Upstox JSON numeric unit/scale remains independently verified before live use |
| Corporate actions | Upstox Corporate Actions API returns Dividend, Bonus, Split and Rights events by ISIN with ex/effective date and ratio/amount | Current docs | https://upstox.com/developer/api-documentation/get-corporate-actions/ | VERIFIED |
| NSE MII CM security file | NSE publishes `NSE_CM_security_ddmmyyyy.csv.gz` daily on the website | 2024-02-05 onward | https://nsearchives.nseindia.com/content/circulars/MSD60315.pdf | VERIFIED; point-in-time files exist from this date |
| NSE Master Data v1.5 date | NSE Master Data Specification v1.5 is dated 01-Jul-2024 | 2024-07-01 evidence boundary | https://nsearchives.nseindia.com/web/sites/default/files/inline-files/NSE_MasterData_Technical_Specifications.pdf | VERIFIED; first exact semantics boundary used by this engine |
| NSE CM price units | Currency INR; CM price fields are in paise and must be divided by 100 to obtain rupees | v1.5, conservatively from 2024-07-01 | same Master Data v1.5 source | VERIFIED |
| NSE CM instrument type | Instrument type 0 = equities; 1 preference; 2 debentures; 3 warrants; 4 miscellaneous | v1.5, conservatively from 2024-07-01 | same Master Data v1.5 source | VERIFIED |
| NSE CM permitted-to-trade | 0 = listed but not permitted; 1 = permitted | v1.5, conservatively from 2024-07-01 | same Master Data v1.5 source | VERIFIED |
| NSE CM market eligibility | Eligibility 1 = allowed to trade; 0 = not allowed | v1.5, conservatively from 2024-07-01 | same Master Data v1.5 source | VERIFIED |
| NSE CM security status | 1 preopen, 2 open, 3 suspended, 4 preopen extended, 5 stock-open-with-market, 6 price discovery | v1.5, conservatively from 2024-07-01 | same Master Data v1.5 source | VERIFIED; day-level gate treats explicit suspension 3 as blocked and still requires permission + eligibility |
| Current NSE permitted code extension | Current 2026 CM NNF additionally documents code 2 = BSE-listed exclusive security, only tradable on NSE during BSE outage | Current 2026 protocol | https://nsearchives.nseindia.com/web/mediaattachment/2026-02/TP_EGR_Trimmed_NNF_PROTOCOL_1_1_20260210174632.pdf | VERIFIED CURRENT; not back-projected into v1.5 historical semantics |
| NSE MII SSEC field | Annexure 10 field 11 uses ISO tag `CallAuctnInd`; NSE stated no other file-structure/format change in that modification | 2025-06-16 | https://nsearchives.nseindia.com/content/circulars/CMTR68176.pdf | VERIFIED |
| NSE MII archive implementation path | Working open-source NSE data implementation uses `/content/cm/NSE_CM_security_{ddmmyyyy}.csv.gz` and parses it as gzip CSV | Current implementation reference | https://github.com/Praglitch/market-data-feed/blob/main/packages/nse-data/src/nsedata/registry.py | CORROBORATED IMPLEMENTATION REFERENCE; not regulatory authority |
| NSE MII ISO-tag schema cross-check | Independent parser recognizes `FinInstrmId`, `TckrSymb`, `SctySrs`, `ISIN`, `NewBrdLotQty`, `SctyTpFlg`, `BidIntrvl`, `CallAuctnInd`, `PrtdToTrad`, `SctyStsNrmlMkt`, `ElgbltyNrmlMkt` and schema changes around CAS | Current implementation reference | https://github.com/KamalSuman/india-swing-trading-system/blob/main/src/india_swing/reference_data/security_master.py | CORROBORATED IMPLEMENTATION REFERENCE; parser fails closed on missing tags/unknown semantics |
| NSE security-file tick authority | NSE states the applicable trading tick size is available in the daily security files | Current applicable rule | https://nsearchives.nseindia.com/content/circulars/CMTR67133.pdf | VERIFIED |
| NSE CM tick-size tiers | Below ₹250: ₹0.01; ₹250-₹1,000: ₹0.05; >₹1,000-₹5,000: ₹0.10; >₹5,000-₹10,000: ₹0.50; >₹10,000-₹20,000: ₹1.00; >₹20,000: ₹5.00 | 2025-04-15 onward, monthly review | https://nsearchives.nseindia.com/content/circulars/CMTR67133.pdf | VERIFIED |
| NSE cash transaction charge | ₹306.99 per crore each side | 2026-03-01 | https://nsearchives.nseindia.com/content/circulars/FA73061.pdf | VERIFIED |
| NSE cash IPFT contribution | ₹0.01 per crore each side | 2026-03-01 | https://nsearchives.nseindia.com/content/circulars/FA73061.pdf | VERIFIED |
| NSE cash combined exchange outflow | ₹307.00 per crore each side | 2026-03-01 | https://nsearchives.nseindia.com/content/circulars/FA73061.pdf | VERIFIED; model keeps transaction and IPFT separate |
| Intraday STT | 0.025% on sell side for non-delivery equity sale | Current | https://www.nseindia.com/static/invest/first-time-investor-sebi-turnover-fees-stt-other-levies | VERIFIED |
| NSE CAS applicability | Phase 1: cash stocks on which derivative contracts are available | 2026-08-03 | https://www.nseindia.com/static/products-services/closing-auction-session | VERIFIED |
| NSE CAS timing | CAS securities leave continuous session at 15:15; CAS 15:15-15:35. Non-CAS cash securities continue to 15:30 | 2026-08-03 | https://www.nseindia.com/static/products-services/closing-auction-session | VERIFIED |
| Trade-to-Trade surveillance | NSE reviews movement into Trade-to-Trade fortnightly and movement in/out quarterly | 2026-2027 calendar | https://www.nseindia.com/static/regulations/exchange-market-surveillance-actions-tt-schedule | VERIFIED; historical research needs point-in-time exchange evidence |
| VectorBT anti-lookahead | Signals generated using close should be shifted forward; execute at a price after the signal | Current docs | https://vectorbt.dev/api/portfolio/base/ | VERIFIED |
| VectorBT version | 1.1.0 | Released 2026-07-05 | https://pypi.org/project/vectorbt/ | VERIFIED |
| VectorBT dependency floor | Actual install metadata requires NumPy >=2.4.6 and pandas >=3.0.3,<4 | VectorBT 1.1.0 | GitHub Actions resolver + package metadata | VERIFIED by CI |

## Effective-dated NSE semantics policy

`nse_cm_master_data_v15_semantics()` is anchored at **2024-07-01**, the date printed on NSE Master Data Specification v1.5. The engine deliberately does not claim that this document proves identical semantics for February-June 2024.

For that dated contract:

- configured normal-equity research series is `EQ`;
- `PrtdToTrad=0` means the security is listed but cannot be traded;
- `PrtdToTrad=1` is required for a normal entry;
- normal-market `ElgbltyNrmlMkt=1` is required;
- status `3` is explicit suspension and blocks trading;
- documented states `1`, `2`, `4`, `5`, `6` are treated as active day-level states, while the actual OHLCV layer separately proves that market bars existed for the day;
- CM raw price fields use scale `0.01` rupees per paise unit.

Unknown codes fail. A current code such as `PrtdToTrad=2` cannot be interpreted by the 2024 v1.5 contract; a new effective-dated contract must be added from an authoritative source.

## Intentionally unresolved / not assumed

### Historical semantics from 2024-02-05 through 2024-06-30

NSE/MSD/60315 verifies website dissemination of the MII daily master from 05-Feb-2024, but the primary Master Data v1.5 semantics document we have locked is dated 01-Jul-2024. Therefore 05-Feb through 30-Jun remains usable for raw point-in-time file acquisition but is not labelled exact-semantic research yet.

### Historical universe before 2024-02-05

We do not infer the 2022-2024-02-04 point-in-time universe from today's broker list. Older candle data may be used for exploratory strategy behavior only until a verified legacy daily security-master source is incorporated.

### MII `BidIntrvl` field-name bridge

Primary NSE Master Data v1.5 verifies CM price values are paise and the security master has a numeric `Tick Size` / minimum-spread field. The MII ISO-tag parser and independent implementations identify `BidIntrvl` as the corresponding bid/tick interval. The engine therefore records both the raw MII field and the primary CM `/100` unit source in every tick lineage. Any later contradiction between MII and exchange tick-tier/security-master evidence fails reconciliation rather than being silently repaired.

### Upstox JSON tick-size numeric scale

Upstox JSON documentation describes `tick_size` as minimum price movement but does not explicitly state the JSON numeric scale. Current paper/live eligibility therefore cross-checks converted Upstox tick size against exchange evidence; it does not use the Upstox field as historical authority.

### Historical-candle corporate-action adjustment

The verified Upstox V3 candle documentation does not state whether OHLC history is adjusted for splits, bonuses or rights. Upstox exposes those events separately through the Corporate Actions API. Until an adjustment/normalization method is verified, research windows containing caller-designated structural events are blocked rather than silently adjusted.

### Historical statutory and broker fee schedule for 2022-2025

The current `CurrentTermsNSEIntradayCostProvider` answers: would a historical signal survive the **current verified cost schedule**? It does not claim to recreate the fee schedule that applied on every historical trade date. An effective-dated historical fee ledger is still required before a report is labelled `HISTORICAL_ACTUAL_COSTS`.

### Delivery / swing exact cost

Swing/delivery live eligibility is blocked until DP/demat charges and all delivery-specific components are represented and reconciled against the user's actual broker DP plan.

### Historical liquidity vs exact turnover

OHLCV does not contain every execution price, so the research universe uses the clearly labelled daily notional proxy `sum(bar close × bar volume)`. It is not called exact exchange turnover.

### Spread and slippage

No universal spread/slippage number is hard-coded. Historical OHLC is not used as a bid/ask-spread substitute. Paper/live eligibility separately requires observed bid/ask samples.

### NSE CAS participation

V1 deliberately avoids placing research/live orders in CAS. The session policy exits before the applicable continuous-market end using an explicit caller-supplied buffer. CAS auction-fill modelling is separate research.

### Special trading sessions

Muhurat/special-session times are not guessed. `NSEEquitySessionPolicy` accepts explicit date overrides sourced from exchange notices.
