# Historical Pilot Evidence Pack V1

- **Mode:** DRY_RUN only; no historical candles were downloaded.
- **Selection:** deterministic dated NSE/PIT intersection + canonical dated CAS eligibility, sorted by `instrument_key`, first result; current BOD is nonblocking corroborative metadata and no price/performance ranking was used.
- **Selected instrument:** `NSE_EQ|INE745G01043` (`MCX`)
- **Pilot dates:** `2026-07-30`, `2026-07-31`, `2026-08-03` (`NSE_CAS_EFFECTIVE_DATE`; CAS-aware)
- **Special-session audit example:** `2026-11-08`, `OUTSIDE_PILOT_WINDOW` / `NOT_COUNTED` / `SPECIAL_SESSION_EXCLUDED`; not part of the canonical acquisition window and not downloaded.
- **Instrument-trading-days:** `3` (limit 7)
- **Acquisition requests:** `1`
- **Estimated rows:** `226` (`75 + 75 + 76`, including the CAS auxiliary upper bound)
- **Estimated raw storage:** `36160` bytes at `160` bytes/row
- **Canonical acquisition plan:** `hap_f7dd6582ff582a3a` / `f7dd6582ff582a3aecadb77172ad06b5a99ce9df30171489f5fab0603c48f795`
- **Acquisition evidence fingerprint:** `7e8aa99b74ac032dbb0c6b104d1867182c2a34ad3811a96b0293ead5136dfc6e`
- **Aggregate PIT evidence fingerprint:** `71bb39108ec5ffb4b17ded22b79aecee0ef7fd32e05b285f47d17c63af852d37` (all three canonical PIT segments)
- **Corporate-action status:** complete query coverage; `NO_ACTION_CONFIRMED_BY_COMPLETE_COVERAGE`; UNKNOWN is not treated as no action.
- **Session:** `Asia/Kolkata`, continuous `09:15`–`15:30` on normal dates; CAS continuous end `15:15`, auxiliary `15:15`–`15:35` on `2026-08-03`; source `https://www.nseindia.com/static/products-services/closing-auction-session`.
- **Safety:** `live_orders_called=false`; credentials are not present in the pack.

The special-session record is intentionally a separately sourced audit example: it is outside the pilot window, marked `OUTSIDE_PILOT_WINDOW` and `NOT_COUNTED`, and is not treated as an in-window acquisition exclusion.
