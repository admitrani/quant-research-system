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


def run_ingestion():

    if get_last_timestamp(SYMBOL, INTERVAL) is None:
        bootstrap_start = datetime(2020, 1, 1)
    else:
        bootstrap_start = None

    df = fetch_klines(
        symbol=SYMBOL,
        interval=INTERVAL,
        start_date=bootstrap_start
    )

    if df.empty:
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
        logger.warning("Duplicates detected in raw data.")

    if report["gaps"] > 0:
        logger.warning("Temporal gaps detected in raw data.")


def run_silver_transformations():
    
    logger.info("Materializing Silver layer...")

    con = duckdb.connect("warehouse.duckdb")

    # Create raw view
    raw_sql = Path("transformations/models/raw/raw_prices.sql").read_text()
    con.execute(f"CREATE OR REPLACE VIEW raw_prices AS {raw_sql}")

    # Remove previous Silver table
    con.execute("DROP TABLE IF EXISTS silver_prices")

    # Create materialized Silver table
    silver_sql = Path("transformations/models/silver/silver_prices.sql").read_text()
    con.execute(f"CREATE TABLE silver_prices AS {silver_sql}")

    # Validate transformation checks
    logger.info("Running Silver validation...")
    from transformations.validators.silver_validator import validate_silver_prices
    validate_silver_prices(con)

    # Remove previous Silver parquet
    silver_storage_path = "storage/silver/prices.parquet"
    if os.path.exists(silver_storage_path):
        os.remove(silver_storage_path)

    # Export Silver to Parquet
    con.execute(f"""
        COPY silver_prices
        TO '{silver_storage_path}'
        (FORMAT PARQUET);
    """)

    con.close()

    logger.info("Silver layer materialized successfully.")


def run_gold_transformations():
    pass


def run_model_stage():
    pass

