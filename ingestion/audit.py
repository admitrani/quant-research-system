from pathlib import Path
import pandas as pd
from datetime import datetime


def load_all_timestamps(base_path):
    parquet_files = list(base_path.rglob("*.parquet"))

    if not parquet_files:
        return pd.Series(dtype="int64")

    ts_parts = []

    for file in parquet_files:
        df = pd.read_parquet(file, columns=["0"])
        col = pd.to_numeric(df["0"], errors="coerce").dropna()
        if not col.empty:
            ts_parts.append(col.astype("int64"))

    if not ts_parts:
        return pd.Series(dtype="int64")

    timestamps = pd.concat(ts_parts, ignore_index=True)
    return timestamps.sort_values().reset_index(drop=True)


def basic_raw_checks(base_path, interval_ms):
    timestamps = load_all_timestamps(base_path)

    if timestamps.empty:
        return {
            "total_candles": 0,
            "duplicates": 0,
            "is_sorted": True,
            "gaps": 0,
            "missing_candles": 0,
        }

    total = len(timestamps)

    # Duplicados
    duplicates = timestamps.duplicated().sum()

    # Orden
    is_sorted = timestamps.is_monotonic_increasing

    # Continuidad
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

    parquet_files = list(base_path.rglob("*.parquet"))

    if not parquet_files:
        return pd.DataFrame()

    timestamps = []

    for file in parquet_files:
        df = pd.read_parquet(file, columns=["0"])
        timestamps.append(pd.to_numeric(df["0"], errors="coerce"))

    ts = pd.concat(timestamps).dropna().astype("int64")
    ts = ts.drop_duplicates().sort_values().reset_index(drop=True)

    if len(ts) < 2:
        return pd.DataFrame()

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