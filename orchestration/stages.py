from datetime import datetime
from ingestion.market_data import fetch_klines, save_raw, get_last_timestamp
from ingestion.audit import basic_raw_checks
from pathlib import Path
import logging
import duckdb
import os


SYMBOL = "BTCUSDT"
INTERVAL = "1h"


logger = logging.getLogger(__name__)


def run_ingestion(backfill_start=None):

    logger.info("Starting ingestion...")

    from ingestion.utils import get_partitions_from_date

    if backfill_start:
        logger.info(f"Running backfill from: {backfill_start}")

        start_dt = datetime.fromisoformat(backfill_start)
        partitions = get_partitions_from_date(start_dt)

        for year, month in partitions:
            partition_path = Path(
                f"storage/raw/source=binance/dataset=klines/"
                f"symbol={SYMBOL}/interval={INTERVAL}/year={year}/month={month}"
            )

            if partition_path.exists():
                for file in partition_path.glob("*.parquet"):
                    file.unlink()
                cleared_count += 1
                logger.info(f"Cleared {cleared_count} partitions for backfill.")
    
        df = fetch_klines(
            symbol=SYMBOL,
            interval=INTERVAL,
            start_date=start_dt
        )

    else:
        if get_last_timestamp(SYMBOL, INTERVAL) is None:
            bootstrap_start = datetime(2020, 1, 1)
        else:
            bootstrap_start = None
        
        df = fetch_klines(
            symbol=SYMBOL,
            interval=INTERVAL,
            start_date=bootstrap_start
        )

    new_rows = len(df)
    logger.info(f"New rows this run: {new_rows}")
    had_new_data = new_rows > 0

    #Runtime duplicate check on this batch
    if new_rows > 0:
        batch_duplicates = df["0"].duplicated().sum()

        logger.info(f"Duplicates in current batch: {batch_duplicates}")

        if batch_duplicates > 0:
            logger.error("Duplicates detected in ingestion batch.")
            raise Exception("Raw batch duplicate violation")

    #Runtime sanity check on volume of new data
    if new_rows > 1000: #adjustable
        logger.warning(f"Unusually high number of new candles ingested.")

    if new_rows == 0:
        logger.info("Nothing to ingest.")
    else:
        save_raw(df, symbol=SYMBOL, interval=INTERVAL)
        logger.info(f"Ingested {len(df)} candles.")

    base_path = Path(
        f"storage/raw/source=binance/dataset=klines/"
        f"symbol={SYMBOL}/interval={INTERVAL}"
    )

    interval_ms = 3600000

    report = basic_raw_checks(base_path, interval_ms)
    
    logger.info("Raw ingestion summary:")
    logger.info(f"Total candles: {report['total_candles']}")
    logger.info(f"Duplicates: {report['duplicates']}")
    logger.info(f"Is sorted: {report['is_sorted']}")
    logger.info(f"Gaps detected: {report['gaps']}")
    logger.info(f"Missing candles: {report['missing_candles']}")

    if report["duplicates"] > 0:
        logger.error("Duplicates detected in raw data.")
        raise Exception("Raw duplicate violation")

    if report["gaps"] > 0:
        logger.warning("Temporal gaps detected in raw data.")
        logger.warning(f"Missing candles: {report['missing_candles']}")

    if backfill_start:
        if report["gaps"] > 0:
            logger.error("Backfill introduced historical gaps.")
            raise Exception("Backfill continuity violation")
    
    return had_new_data


def run_silver_transformations():
    
    logger.info("Materializing Silver layer...")

    con = duckdb.connect("warehouse.duckdb")

    try:
        # Create raw view
        raw_sql = Path("transformations/models/raw/raw_prices.sql").read_text()
        con.execute(f"CREATE OR REPLACE VIEW raw_prices AS {raw_sql}")

        # Remove previous Silver table
        con.execute("DROP TABLE IF EXISTS silver_prices")

        # Create materialized Silver table
        silver_sql = Path("transformations/models/silver/silver_prices.sql").read_text()
        con.execute(f"CREATE TABLE silver_prices AS {silver_sql}")

        # Validate model
        logger.info("Running Silver validation...")
        from transformations.validators.silver_validator import validate_silver_prices
        validate_silver_prices(con)

        # Runtime null ratio check
        logger.info("Running runtime null ratio checks...")
        row_count = con.execute("SELECT COUNT(*) FROM silver_prices").fetchone()[0]
        null_close = con.execute("SELECT COUNT(*) FROM silver_prices WHERE close_price IS NULL").fetchone()[0]

        NULL_THRESHOLD = 0.0 #adjustable
    
        if row_count > 0:
            null_ratio = null_close / row_count
            logger.info(f"runtime null ratio (close_price): {null_ratio:.6f}")

            if null_ratio > NULL_THRESHOLD:
                logger.error("Null close_price detected at runtime.")
                raise Exception("Runtime null violation")
            
        #Runtime volume anomaly check
        logger.info("Running runtime volume anomaly check...")

        stats = con.execute("""SELECT AVG(volume), STDDEV(volume), MAX(volume) FROM silver_prices""").fetchone()

        avg_vol, std_vol, max_vol = stats

        if avg_vol is not None and std_vol is not None and std_vol > 0:
            z_score = (max_vol - avg_vol) / std_vol

            logger.info(f"Max volume z-score: {z_score:.2f}")

            if z_score > 200:
                logger.error("Extreme volume anomaly detected.")
                raise Exception("Volume anomaly error")
            
            elif z_score > 50:
                logger.warning("High volume anomaly detected.")

        # Remove previous Silver parquet
        silver_storage_path = "storage/silver/prices.parquet"
        if os.path.exists(silver_storage_path):
            os.remove(silver_storage_path)

        # Export Silver to Parquet
        con.execute(f"""COPY silver_prices TO '{silver_storage_path}' (FORMAT PARQUET)""")

        logger.info("Silver layer materialized successfully.")

    finally:
        con.close()


def run_gold_transformations():
    pass


def run_model_stage():
    pass

