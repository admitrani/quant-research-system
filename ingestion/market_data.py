from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from ingestion.api_client import APIClient
import logging

from config.config_loader import load_config

BASE_URL = "https://api.binance.com"

logger = logging.getLogger(__name__)

INTERVAL_TO_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}

def interval_to_milliseconds(interval):
    if interval not in INTERVAL_TO_MS:
        logger.error(f"Unsupported interval: {interval}")
        raise ValueError(f"Unsupported interval: {interval}")
    return INTERVAL_TO_MS[interval]

# GET LAST TIMESTAMP (Watermark from Raw)

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


def fetch_klines(
    symbol="BTCUSDT",
    interval="1h",
    days=30,
    start_date=None,
    end_time_ms=None
):

    config = load_config()
    ingestion_config = config["market_data"]["ingestion"]  

    client = APIClient(
        base_url=BASE_URL,
        max_retries=ingestion_config["max_retries"],
        backoff_seconds=ingestion_config["backoff_seconds"]
    )

    interval_ms = interval_to_milliseconds(interval)
    limit = ingestion_config["limit"]
    last_ts = get_last_timestamp(symbol, interval)

    if start_date:
        start_time = int(start_date.timestamp() * 1000)

    elif last_ts is None:
        start_time = int(
            (datetime.utcnow() - timedelta(days=days)).timestamp() * 1000
        )

    else:
        start_time = int(last_ts) + interval_ms

    all_data = []

    while True:

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "limit": limit
        }

        if end_time_ms:
            params["endTime"] = end_time_ms

        data = client.get("/api/v3/klines", params=params)

        if not data:
            break

        all_data.extend(data)

        last_open_time = data[-1][0]

        if end_time_ms and last_open_time >= end_time_ms:
            break

        start_time = last_open_time + interval_ms

        if len(data) < limit:
            break

    if not all_data:
        logger.info("No new data to fetch.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df.columns = df.columns.astype(str)

    return df


def save_raw(df, symbol="BTCUSDT", interval="1h"):

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
            
        timestamp_str = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        file_path = base_path / f"data_{timestamp_str}.parquet"

        group.to_parquet(file_path, index=False)