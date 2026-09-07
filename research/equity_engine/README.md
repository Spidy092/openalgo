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
