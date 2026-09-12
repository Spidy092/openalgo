# Historical Pilot Evidence Pack V1

- **Mode:** DRY_RUN only; no historical candles were downloaded.
- **Selection:** deterministic PIT intersection + CAS eligibility, sorted by `instrument_key`, first result; no price/performance ranking.
- **Selected instrument:** `NSE_EQ|INE745G01043` (`MCX`)
- **Pilot dates:** `2026-07-30`, `2026-07-31`, `2026-08-03` (`NSE_CAS_EFFECTIVE_DATE`; CAS-aware)
- **Planner-level excluded special session:** `2026-11-08`, `SPECIAL_SESSION_EXCLUDED`; not counted or downloaded.
- **Instrument-trading-days:** `3` (limit 7)
- **Acquisition requests:** `1`
- **Estimated rows:** `226` (`75 + 75 + 76`, including the CAS auxiliary upper bound)
- **Estimated raw storage:** `36160` bytes at `160` bytes/row
- **Canonical acquisition plan:** `hap_5f45d0b05f1b1ffa` / `5f45d0b05f1b1ffa067a9b33b558bbf134a5f4f2caee2695cfe6dffd80fd5fb3`
- **Acquisition evidence fingerprint:** `b93d8c131f0037ff5f8b9880d3a74d532f09a28fdebce612512c721184a24c85`
- **Corporate-action status:** complete query coverage; `NO_ACTION_CONFIRMED_BY_COMPLETE_COVERAGE`; UNKNOWN is not treated as no action.
- **Session:** `Asia/Kolkata`, continuous `09:15`–`15:30` on normal dates; CAS continuous end `15:15`, auxiliary `15:15`–`15:35` on `2026-08-03`; source `https://www.nseindia.com/static/products-services/closing-auction-session`.
- **Safety:** `live_orders_called=false`; credentials are not present in the pack.

The special-session record is intentionally planner-level because the acquisition window ends at the actual CAS-aware date. It remains explicit and excluded rather than being treated as a normal session.
