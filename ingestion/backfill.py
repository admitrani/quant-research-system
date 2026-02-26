from datetime import datetime, timezone
from ingestion.market_data import fetch_klines, save_raw, interval_to_milliseconds
from ingestion.audit import detect_raw_gaps_from_path
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def run_gap_backfill(symbol, interval):

    logger.info("Running backfill...")

    base_path = Path(f"storage/raw/source=binance/dataset=klines/"f"symbol={symbol}/interval={interval}")
    interval_ms = interval_to_milliseconds(interval)

    gaps = detect_raw_gaps_from_path(base_path, interval_ms)

    if gaps.empty:
        logger.info("No gaps detected. Nothing to backfill.")
        return 0

    total_fetched = 0

    logger.info(f"Gaps detected: {len(gaps)}")

    for gap in gaps.itertuples(index=False):

        gap_start_ms = gap.prev_open_time + interval_ms
        gap_end_ms = gap.next_open_time - interval_ms

        start_dt = datetime.fromtimestamp(gap_start_ms / 1000, tz=timezone.utc)

        logger.debug(
            f"Backfilling gap between " 
            f"{datetime.fromtimestamp(gap.prev_open_time / 1000, tz=timezone.utc)} " 
            f"and "
            f"{datetime.fromtimestamp(gap.next_open_time / 1000, tz=timezone.utc)}")


        df = fetch_klines(
            symbol=symbol,
            interval=interval,
            start_date=start_dt,
            end_time_ms=gap_end_ms,
        )

        if not df.empty:
            save_raw(df, symbol=symbol, interval=interval)
            total_fetched += len(df)

    logger.info(f"Backfill completed. Inserted {total_fetched} candles.")

    return total_fetched