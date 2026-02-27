import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import logging

from ingestion.market_data import fetch_klines, save_raw, interval_to_milliseconds

logger = logging.getLogger(__name__)


def run_range_reprocess(symbol, interval, start_date, end_date):

    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)

    interval_ms = interval_to_milliseconds(interval)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    base_path = Path(
        f"storage/raw/source=binance/dataset=klines/"
        f"symbol={symbol}/interval={interval}"
    )

    logger.info("Rewriting affected files...")

    for file in base_path.rglob("*.parquet"):

        df = pd.read_parquet(file)
        mask = (df["0"] >= start_ms) & (df["0"] <= end_ms)

        if mask.any():
            df_filtered = df[~mask]
            if df_filtered.empty:
                logger.debug(f"Deleting emptyfile: {file}")
                file.unlink()
            else:
                logger.debug(f"Rewriting file: {file}")
                df_filtered.to_parquet(file, index=False)

    logger.info("Fetching fresh data for range...")

    df = fetch_klines(
        symbol=symbol,
        interval=interval,
        start_date=start_dt,
        end_time_ms=end_ms
    )

    if df.empty:
        logger.warning("No data returned during reprocess.")
        return 0

    save_raw(df, symbol=symbol, interval=interval)

    return len(df)