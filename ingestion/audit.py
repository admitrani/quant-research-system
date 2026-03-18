from pathlib import Path
import pandas as pd
from datetime import datetime


def _load_raw_timestamps(base_path: Path) -> pd.Series:
    """Load and concatenate open_time timestamps from all raw parquet files."""
    parquet_files = list(base_path.rglob("*.parquet"))

    if not parquet_files:
        return pd.Series(dtype="int64")

    parts = []

    for file in parquet_files:
        df = pd.read_parquet(file, columns=["0"])
        col = pd.to_numeric(df["0"], errors="coerce").dropna().astype("int64")
        if not col.empty:
            parts.append(col)

    if not parts:
        return pd.Series(dtype="int64")

    return pd.concat(parts, ignore_index=True).sort_values().reset_index(drop=True)


def load_all_timestamps(base_path):
    """Public wrapper — kept for backward compatibility."""
    return _load_raw_timestamps(base_path)


def basic_raw_checks(base_path, interval_ms):
    timestamps = _load_raw_timestamps(base_path)

    if timestamps.empty:
        return {
            "total_candles": 0,
            "duplicates": 0,
            "is_sorted": True,
            "gaps": 0,
            "missing_candles": 0,
        }

    total = len(timestamps)

    # Duplicates
    duplicates = timestamps.duplicated().sum()

    # Order
    is_sorted = timestamps.is_monotonic_increasing

    # Continuation
    diffs = timestamps.diff().dropna()
    gaps_series = diffs[diffs > interval_ms]

    gaps = len(gaps_series)

    missing_candles = int((gaps_series / interval_ms - 1).sum())

    return {
        "total_candles": int(total),
        "duplicates": int(duplicates),
        "is_sorted": bool(is_sorted),
        "gaps": int(gaps),
        "missing_candles": int(missing_candles),
    }


def detect_raw_gaps_from_path(base_path, interval_ms):

    ts = _load_raw_timestamps(base_path)

    if ts.empty or len(ts) < 2:
        return pd.DataFrame()

    ts = ts.drop_duplicates().reset_index(drop=True)

    continuity = pd.DataFrame({"next_open_time": ts})
    continuity["prev_open_time"] = continuity["next_open_time"].shift(1)
    continuity["diff_ms"] = continuity["next_open_time"] - continuity["prev_open_time"]

    gaps = continuity[continuity["diff_ms"] > interval_ms].copy()

    if gaps.empty:
        return pd.DataFrame()

    gaps["missing_candles"] = (gaps["diff_ms"] // interval_ms - 1).astype("int64")

    return gaps[[
        "prev_open_time",
        "next_open_time",
        "missing_candles"
    ]]
