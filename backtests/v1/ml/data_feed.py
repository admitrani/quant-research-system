import pandas as pd
from pathlib import Path
import backtrader as bt

from config.config_loader import get_gold_path

def load_backtest_dataset():
    """
    Load the complete Gold v1 dataset for backtesting.
    
    Returns the full dataset from 2019 without filtering by backtest_start.
    Temporal filtering occurs implicitly downstream when signal_engine.py
    calls dropna(subset=["ml_prob"]) — only rows with ML predictions
    (i.e., OOS windows starting from backtest_start) are retained.
    
    Returns
    -------
    pd.DataFrame
        Full Gold v1 dataset with OHLCV, features, and labels.
    """
    gold_path = get_gold_path()

    df = pd.read_parquet(gold_path)

    df["open_time_utc"] = pd.to_datetime(df["open_time_utc"])
    df = df.sort_values("open_time_utc")
    df.set_index("open_time_utc", inplace=True)

    return df


class MLDataFeed(bt.feeds.PandasData):

    lines = ("ml_prob",)

    params = (
        ("datetime", None),
        ("open", "open_price"),
        ("high", "high_price"),
        ("low", "low_price"),
        ("close", "close_price"),
        ("volume", "volume"),
        ("openinterest", -1),
        ("ml_prob", "ml_prob"),
    )
