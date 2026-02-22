from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from ingestion.api_client import APIClient

BASE_URL = "https://api.binance.com"


# GET LAST TIMESTAMP (Watermark real desde Raw)

def get_last_timestamp(symbol="BTCUSDT", interval="1h"):

    base_path = Path(
        f"storage/raw/source=binance/dataset=klines/"
        f"symbol={symbol}/interval={interval}"
    )

    if not base_path.exists():
        return None

    parquet_files = list(base_path.rglob("*.parquet"))

    if not parquet_files:
        return None

    max_timestamp = None

    for file in parquet_files:
        df = pd.read_parquet(file, columns=["0"])
        current_max = df["0"].max()

        if max_timestamp is None or current_max > max_timestamp:
            max_timestamp = current_max

    return max_timestamp


# FETCH KLINES (Incremental + Paginación)

def fetch_klines(
    symbol="BTCUSDT",
    interval="1h",
    days=30,
    start_date=None
):

    client = APIClient(BASE_URL)

    interval_ms = 3600000  # 1h
    limit = 1000
    end_time = int(datetime.utcnow().timestamp() * 1000)
    last_ts = get_last_timestamp(symbol, interval)

    # Determinar start_time

    if start_date:
        start_time = int(start_date.timestamp() * 1000)

    elif last_ts is None:
        start_time = int(
            (datetime.utcnow() - timedelta(days=days)).timestamp() * 1000
        )

    else:
        start_time = int(last_ts) + interval_ms

    all_data = []

    # Loop de paginación

    while True:

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "limit": limit
        }

        data = client.get("/api/v3/klines", params=params)

        if not data:
            break

        all_data.extend(data)

        last_open_time = data[-1][0]

        start_time = last_open_time + interval_ms

        if len(data) < limit:
            break

    if not all_data:
        print("No new data.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df.columns = df.columns.astype(str)

    return df


# SAVE RAW (Particionado por fecha real + Append inteligente)

def save_raw(df, symbol="BTCUSDT", interval="1h"):

    # Derivar fecha real desde open_time
    df["_open_time_dt"] = pd.to_datetime(df["0"], unit="ms")
    df["_year"] = df["_open_time_dt"].dt.year
    df["_month"] = df["_open_time_dt"].dt.month.apply(lambda x: f"{x:02d}")

    grouped = df.groupby(["_year", "_month"])

    for (year, month), group in grouped:

        base_path = Path(
            f"storage/raw/source=binance/dataset=klines/"
            f"symbol={symbol}/interval={interval}/"
            f"year={year}/month={month}"
        )

        base_path.mkdir(parents=True, exist_ok=True)

        group = group.drop(columns=["_open_time_dt", "_year", "_month"])

        # nombre único por ejecución
        timestamp_str = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

        file_path = base_path / f"data_{timestamp_str}.parquet"

        group.to_parquet(file_path, index=False)