import pandas as pd
import numpy as np


def create_mock_dataset(seed=42):

    np.random.seed(seed)

    # 5 years hourly data (≈ 43800 rows)
    dates = pd.date_range(
        start="2018-01-01",
        end="2023-01-01",
        freq="h"
    )

    n = len(dates)

    # Random walk price
    price = 10000 + np.cumsum(np.random.normal(0, 20, n))

    df = pd.DataFrame({
        "open_time_utc": dates,
        "open_price": price + np.random.normal(0, 5, n),
        "high_price": price + np.random.normal(5, 5, n),
        "low_price": price - np.random.normal(5, 5, n),
        "close_price": price,
        "volume": np.random.normal(100, 10, n)
    })

    df.set_index("open_time_utc", inplace=True)

    # Features
    df["return_1h"] = df["close_price"].pct_change()
    df["return_3h"] = df["close_price"].pct_change(3)
    df["return_12h"] = df["close_price"].pct_change(12)

    df["volatility_12h"] = df["return_1h"].rolling(12).std()

    ma20 = df["close_price"].rolling(20).mean()
    df["ma20_distance"] = df["close_price"] / ma20 - 1

    vol_mean = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std()
    df["volume_zscore"] = (df["volume"] - vol_mean) / vol_std

    df["future_return"] = df["close_price"].shift(-3) / df["close_price"] - 1

    df["label"] = (df["future_return"] > 0).astype(int)

    df = df.dropna()

    return df