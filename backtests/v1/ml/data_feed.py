import pandas as pd
from pathlib import Path
import backtrader as bt


def load_backtest_dataset():

    project_root = Path(__file__).resolve().parents[1]
    gold_path = project_root / "storage" / "gold" / "btcusdt_1h_v1.parquet"

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
