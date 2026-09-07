from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class MarketDataManifest:
    provider: str
    exchange: str
    instrument_token: str
    symbol: str
    timezone: str
    interval: str
    start: datetime
    end: datetime
    retrieved_at: datetime
    adjustment_policy: str
    universe_rule_version: str
    source_reference: str


def dataframe_fingerprint(frame: pd.DataFrame, manifest: MarketDataManifest) -> str:
    """Create a deterministic fingerprint from metadata plus dataframe values/index.

    Sorting is deliberately not performed here: order is part of the dataset identity. A caller
    must validate chronological ordering before accepting a dataset.
    """

    if frame.empty:
        raise ValueError("cannot fingerprint an empty dataframe")

    metadata = json.dumps(
        asdict(manifest),
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    hashed_rows = pd.util.hash_pandas_object(frame, index=True).values.tobytes()

    digest = hashlib.sha256()
    digest.update(metadata)
    digest.update(hashed_rows)
    return digest.hexdigest()


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
