from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


FINGERPRINT_SCHEMA = "equity-market-data-v2"


@dataclass(frozen=True)
class MarketDataManifest:
    provider: str
    exchange: str
    instrument_token: str
    symbol: str
    timezone: str
    interval: str
    timestamp_semantics: str
    start: datetime
    end: datetime
    retrieved_at: datetime
    adjustment_policy: str
    universe_rule_version: str
    source_reference: str


# Retrieval time is intentionally absent: it describes when the artifact was fetched, not
# what data the artifact contains. Keep this allow-list explicit so adding a volatile manifest
# field cannot silently change dataset identity.
_FINGERPRINT_MANIFEST_FIELDS = (
    "provider",
    "exchange",
    "instrument_token",
    "symbol",
    "timezone",
    "interval",
    "timestamp_semantics",
    "start",
    "end",
    "adjustment_policy",
    "universe_rule_version",
    "source_reference",
)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def _canonical_fingerprint_payload(
    frame: pd.DataFrame, manifest: MarketDataManifest
) -> dict[str, Any]:
    row_hashes = pd.util.hash_pandas_object(frame, index=True).tolist()
    return {
        "manifest": {
            field: _canonical_value(getattr(manifest, field))
            for field in _FINGERPRINT_MANIFEST_FIELDS
        },
        "frame": {
            "columns": [_canonical_value(column) for column in frame.columns.tolist()],
            "column_dtypes": [str(dtype) for dtype in frame.dtypes.tolist()],
            "index_name": _canonical_value(frame.index.name),
            "index_dtype": str(frame.index.dtype),
            # Keep the ordered row hashes as a JSON list. A set or sorted hash list would make
            # reordered candles look identical, which is unsafe for time-series research.
            "ordered_row_hashes": [int(row_hash) for row_hash in row_hashes],
        },
    }


def dataframe_fingerprint(frame: pd.DataFrame, manifest: MarketDataManifest) -> str:
    """Create a deterministic fingerprint from metadata plus dataframe values/index.

    Sorting is deliberately not performed here: order is part of the dataset identity. A caller
    must validate chronological ordering before accepting a dataset.
    """

    if frame.empty:
        raise ValueError("cannot fingerprint an empty dataframe")

    payload = json.dumps(
        _canonical_fingerprint_payload(frame, manifest),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def validate_ohlcv_frame(frame: pd.DataFrame) -> list[str]:
    """Return data-quality violations; an empty list means structural checks passed."""

    violations: list[str] = []
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        violations.append(f"missing required columns: {sorted(missing)}")
        return violations

    if frame.empty:
        violations.append("dataset is empty")
        return violations

    if not frame.index.is_monotonic_increasing:
        violations.append("timestamps are not monotonically increasing")
    if frame.index.has_duplicates:
        violations.append("duplicate timestamps detected")
    if frame[list(required)].isna().any().any():
        violations.append("OHLCV contains missing values; silent forward-fill is forbidden")
    if (frame["volume"] < 0).any():
        violations.append("negative volume detected")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        violations.append("non-positive price detected")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
        violations.append("OHLC invariant violation: high below another price field")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        violations.append("OHLC invariant violation: low above another price field")

    return violations
