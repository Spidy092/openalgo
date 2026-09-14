# Equity Research Engine

Research-only layer for validating NSE equity strategies before they can reach OpenAlgo execution.

## Safety boundary

This package MUST NOT place live orders. It produces research artifacts, paper-trading signals, cost estimates, and eligibility decisions. Live execution remains behind OpenAlgo/broker controls and a separate manual capital approval.

## Why VectorBT is here

VectorBT is used only for fast hypothesis screening and parameter sweeps. A VectorBT backtest is never sufficient by itself to make a strategy live-eligible.

Research pipeline:

1. Acquire versioned market data and record provenance/fingerprint.
2. Build a dynamic, liquidity-aware equity universe.
3. Generate candidate signals.
4. Run VectorBT for fast parameter/search screening.
5. Include documented fees, taxes, spread/slippage scenarios, and capital constraints.
6. Run train/validation/test splits with the final test period untouched during development.
7. Run rolling walk-forward validation.
8. Re-run survivors through an event-driven fill simulator.
9. Paper trade survivors using real market data.
10. Reconcile estimated transaction costs against the broker's authoritative Brokerage Details API.
11. Require explicit human approval before enabling any live capital.

## Historical batch-data workflow

Historical acquisition boundaries are caller-supplied. NSE CM semantics before the verified
`2024-07-01` evidence boundary are not projected backward. Any CAS or other session change must
be resolved by its effective-dated session policy; it is never treated as an ordinary missing bar.

Build the dated NSE universe:

```bash
python scripts/nse_universe_batch.py \
  --start 2024-07-01 \
  --end 2026-07-31 \
  --output-dir data/nse_universe_batch
```

The batch uses sourced NSE Capital Market holiday calendars, excludes explicitly identified Muhurat/special sessions from the normal-session research pass, caches every raw `NSE_CM_security_DDMMYYYY.csv.gz`, writes a SHA-256 sidecar, materializes one dated Parquet universe per session, and writes `nse_universe_manifest.json`. A missing snapshot on a declared trading date, cache hash mismatch, unknown schema/code, or unresolved eligible duplicate fails closed. Re-running resumes from verified cache. `--refresh-existing` re-downloads cached dates only to verify that the archive payload has not changed; a differing payload is never silently overwritten.

For the two-stage research workflow, first create a dry-run plan from a completed universe
manifest. The command requires capital, affordability/liquidity thresholds, formation/signal/
execution timing, rate-limit evidence, adjustment policy, and storage assumptions explicitly. It
only reads the manifest and writes a deterministic plan; it does not download NSE, Upstox, or
historical candles:

```bash
python scripts/pit_historical_plan.py \
  --universe-manifest data/nse_universe_batch/nse_universe_manifest.json \
  --start "$PIT_START" --end "$PIT_END" \
  --approved-capital 1000 \
  --max-last-price "$MAX_LAST_PRICE" \
  --min-median-daily-notional "$MIN_MEDIAN_DAILY_NOTIONAL" \
  --min-median-daily-volume "$MIN_MEDIAN_DAILY_VOLUME" \
  --min-observed-trading-days "$MIN_OBSERVED_TRADING_DAYS" \
  --min-affordable-quantity 1 \
  --timezone Asia/Kolkata --formation-policy-id pit-prior-close-v1 \
  --decision-time 09:20 \
  --price-reference-policy prior_completed_session_close \
  --signal-time-policy signal-after-formation \
  --execution-time-policy execution-after-signal \
  --rate-limit-policy-id upstox-evidence-rate-limit-v1 \
  --rate-limit-source "$RATE_LIMIT_EVIDENCE" \
  --min-request-interval "$MIN_REQUEST_INTERVAL" \
  --max-attempts 4 --backoff-seconds 1 \
  --universe-rule-version nse-cm-v15-point-in-time \
  --adjustment-policy raw-unadjusted-block-structural-actions \
  --lookback-calendar-days 1 \
  --estimated-rows-per-trading-day 1 \
  --estimated-bytes-per-daily-row 80 \
  --cost-model-identity documented-current-terms-explicit-scenario \
  --output data/pit_historical_acquisition_plan.json
```

Stage A uses the existing resumable Upstox batch downloader at daily resolution after the plan is
approved. Its completed daily datasets are then evaluated by the reusable point-in-time prefilter;
missing observations, malformed OHLCV, missing tick/corporate-action evidence, and unresolved
membership evidence leave the prefilter incomplete. Stage B cannot be built until that artifact is
complete and cryptographically bound to the exact Stage-A plan/source manifest. Only then may the
existing downloader be run at 5-minute resolution for the explicit prefiltered candidates.

The dry-run report deliberately leaves `candidate_count_after_affordability_filter` pending until
Stage-A daily data exists; it never guesses that count from today's universe or prices. Both batch
downloaders use atomic artifact writes, verify cached bytes/hashes on resume, retry only transient
HTTP failures, and set `live_orders_called` to `false`.

Plan the full 5-minute Upstox acquisition without downloading it:

```bash
python scripts/upstox_history_batch.py \
  --universe-manifest data/nse_universe_batch/nse_universe_manifest.json \
  --min-request-interval "$MIN_REQUEST_INTERVAL" \
  --max-attempts 4 --backoff-seconds 1
```

This reports the full-universe candidate count, estimated Upstox request count, estimated rows/storage, and explicitly reports that the approved-capital affordability prefilter has not yet been applied. NSE reference masters do not contain historical market prices, so the program refuses to infer affordability from them.

Actual 5-minute acquisition is deliberately blocked until a separate point-in-time affordability/liquidity step produces an explicit candidate JSON file. Execution requires both `--candidate-file` and `--prefilter-evidence`; it reads `UPSTOX_ACCESS_TOKEN` from the environment, rate-limits/retries transient market-data requests, fingerprints every dataset, saves Parquet + manifest, resumes verified artifacts, and never writes the token to artifacts.

Example candidate-file execution after that prefilter exists:

```bash
python scripts/upstox_history_batch.py \
  --candidate-file data/candidates.json \
  --prefilter-evidence data/candidates.audit.json \
  --min-request-interval "$MIN_REQUEST_INTERVAL" \
  --max-attempts 4 --backoff-seconds 1 \
  --execute
```

Neither batch CLI contains a live-order path.

## Two independent research tracks

### Intraday

- Same-day entry and exit only.
- Current-day market context is part of the signal environment.
- Candidate features include opening gap, NIFTY regime, India VIX, sector relative strength, opening range, VWAP, volume, spread and liquidity.
- Initial real-money experiment, if eventually approved: cash equities only; no F&O, MTF or borrowed capital.

### Swing

- Multi-day holding period, initially bounded to a configured maximum such as 2-10 trading days.
- Daily/weekly market regime, sector strength, corporate-event calendar and liquidity are part of the research context.
- Exit is rule-driven; the position does not wait for the maximum holding period if stop, target or invalidation occurs earlier.

Intraday and swing use separate virtual portfolios so their costs, drawdowns and capital efficiency can be compared honestly.

## No-assumption cost policy

There are two cost sources:

1. `BROKER_QUOTE`: authoritative charge quote returned by the broker API for a concrete instrument/order.
2. `DOCUMENTED_SNAPSHOT`: an effective-dated rate schedule sourced from official broker/exchange/government documentation for historical research.

A documented snapshot is an estimate. It MUST carry source URLs, verification date, effective date and a confidence/reconciliation status. If official documentation is ambiguous, the ambiguous component must not be silently guessed.

Before a strategy can be marked live-eligible, representative orders must be reconciled against the broker's Brokerage Details API. The configured tolerance is expressed in INR and should normally be at paisa-level precision.

Run the read-only cost reconciliation for a supplied NSE instrument token and price:

```bash
export UPSTOX_ACCESS_TOKEN='...'
python scripts/upstox_cost_reconciliation.py \
  --instrument-token 'NSE_EQ|INE002A01018' \
  --symbol RELIANCE \
  --price 250 \
  --pricing-date 2026-09-07 \
  --cost-model documented \
  --tolerance 0.05 \
  --output data/reconciliation/reliance.json
```

The runner generates distinct affordable quantities near ₹100, ₹250, ₹500, ₹750 and ₹950, and checks both BUY and SELL. It uses the broker's reported `charges.total` as authoritative, retains broker charge components in the optional evidence file, uses Decimal-only comparisons, and exits 1 when a difference exceeds the explicitly supplied tolerance. It only calls the read-only Brokerage Details endpoint; live orders are never called. The token is read from the environment and is not printed or persisted.

The safe default `--cost-model documented` uses the public/documented 0.1% brokerage snapshot. The separately selected `--cost-model broker-observed` uses the authenticated 2026-09-09 account snapshot at 0.06% brokerage with component-level paise rounding. The observed snapshot is additional evidence and does not rewrite or retroactively mark earlier documented-model reconciliation results.

## Costs that must be represented

Where applicable:

- brokerage
- STT/CTT
- exchange transaction charges
- SEBI turnover charges
- stamp duty
- GST
- IPFT/IPF charges
- DP/demat charges for delivery exits
- auto-square-off or other broker service charges when applicable
- bid/ask spread
- modeled slippage
- partial-fill impact
- rejected/cancelled-order side effects when relevant

A report that omits applicable costs cannot receive a PASS result.

## Data provenance

Every experiment must record at least:

- provider
- instrument key and exchange
- timezone
- candle interval
- start/end timestamps
- retrieval timestamp
- raw row count
- missing/duplicate candle checks
- adjustment policy for splits/bonuses/dividends
- universe construction rule
- dataset fingerprint

Do not silently forward-fill missing market bars.

## Anti-overfitting policy

A candidate is rejected if any of these are missing:

- held-out test period
- walk-forward evaluation
- costs and slippage stress
- parameter-stability evidence
- enough trades for the metric being interpreted
- comparison with a simple baseline

The highest backtest return is not automatically the winner.

## ₹1,000 experiment

₹1,000 is a capital constraint, not a profit target. The research engine must calculate integer share quantities and reject any trade whose expected edge does not survive all modeled costs and risk gates.

Capital increases are never automatic. A move from ₹1,000 to another amount requires explicit human approval after reviewing net returns, drawdown, cost reconciliation, operational failures and consistency.

## Initial strategy tournament

Candidates to research, not promises of profitability:

- opening-range breakout with liquidity/volume/regime filters
- cross-sectional/instrument momentum with regime filter
- trend pullback
- mean reversion challenger
- simple baseline(s)

Candidates are promoted only by reproducible evidence.

## Integration boundary

```text
Upstox / official market data
          |
          v
research/equity_engine
  - data provenance
  - VectorBT screening
  - cost model
  - walk-forward
  - event-driven validation
  - paper portfolio
          |
          v
research report + signed eligibility decision
          |
          v
OpenAlgo execution layer (later, separately gated)
          |
          v
Upstox
```

The existing `broker/upstox` adapter should not be modified merely to support research.
