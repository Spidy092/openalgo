from datetime import datetime, timezone

import pandas as pd

from equity_engine.provenance import (
    MarketDataManifest,
    dataframe_fingerprint,
    validate_ohlcv_frame,
)


def _frame() -> pd.DataFrame:
    index = pd.date_range("2026-09-01 09:15", periods=3, freq="5min", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.5, 102.5, 103.5],
            "low": [99.5, 100.5, 101.5],
            "close": [101.0, 102.0, 103.0],
            "volume": [1000, 1100, 1200],
        },
        index=index,
    )


def _manifest(frame: pd.DataFrame) -> MarketDataManifest:
    return MarketDataManifest(
        provider="unit-test",
        exchange="NSE",
        instrument_token="NSE_EQ|TEST",
        symbol="TEST",
        timezone="Asia/Kolkata",
        interval="5m",
        timestamp_semantics="candle_start",
        start=frame.index[0].to_pydatetime(),
        end=frame.index[-1].to_pydatetime(),
        retrieved_at=datetime(2026, 9, 7, tzinfo=timezone.utc),
        adjustment_policy="none",
        universe_rule_version="test-v1",
        source_reference="unit-test",
    )


def test_valid_ohlcv_has_no_structural_violations() -> None:
    assert validate_ohlcv_frame(_frame()) == []


def test_duplicate_timestamp_is_rejected() -> None:
    frame = _frame()
    frame = pd.concat([frame, frame.iloc[[-1]]])

    violations = validate_ohlcv_frame(frame)

    assert "duplicate timestamps detected" in violations


def test_missing_values_are_not_silently_forward_filled() -> None:
    frame = _frame()
    frame.loc[frame.index[1], "close"] = None

    violations = validate_ohlcv_frame(frame)

    assert any("missing values" in item for item in violations)


def test_fingerprint_is_stable_and_changes_with_data() -> None:
    frame = _frame()
    manifest = _manifest(frame)

    first = dataframe_fingerprint(frame, manifest)
    second = dataframe_fingerprint(frame.copy(), manifest)
    changed = frame.copy()
    changed.loc[changed.index[-1], "close"] = 103.1
    third = dataframe_fingerprint(changed, manifest)

    assert first == second
    assert first != third
