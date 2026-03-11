from datetime import datetime, timezone
from ingestion.market_data import fetch_klines, save_raw, get_last_timestamp, interval_to_milliseconds
from ingestion.audit import basic_raw_checks
from config.config_loader import load_config
from pathlib import Path
import logging
import duckdb
import os


logger = logging.getLogger(__name__)


def run_ingestion(backfill=False, reprocess_start=None, reprocess_end=None):

    logger.info("Starting ingestion...")

    config = load_config()

    symbol = config["market_data"]["symbols"][0]
    interval = config["market_data"]["intervals"][0]
    historical_start_str = config["market_data"]["historical_start"]

    historical_start = datetime.fromisoformat(historical_start_str).replace(tzinfo=timezone.utc)

    if reprocess_start and reprocess_end:
        logger.info(f"Reprocessing range: {reprocess_start} -> {reprocess_end}")
        from ingestion.reprocess import run_range_reprocess

        inserted = run_range_reprocess(
            symbol,
            interval,
            reprocess_start,
            reprocess_end
        )

        logger.info(f"Reprocess inserted {inserted} candles.")
        return True

    last_ts = get_last_timestamp(symbol, interval)
    bootstrap_start = historical_start if last_ts is None else None
        
    df = fetch_klines(
        symbol=symbol,
        interval=interval,
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
        save_raw(df, symbol=symbol, interval=interval)
        logger.info(f"Ingested {len(df)} candles.")

    base_path = Path(
        f"storage/raw/source=binance/dataset=klines/"
        f"symbol={symbol}/interval={interval}"
    )

    interval_ms = interval_to_milliseconds(interval)

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

    if backfill:
        logger.info("Auto gap backfill enabled.")

        if report["gaps"] > 0:
            from ingestion.backfill import run_gap_backfill
            logger.info(f"Gaps detected before backfill: {report['gaps']}")
            
            inserted = run_gap_backfill(symbol, interval)
            logger.info(f"Candles inserted by backfill: {inserted}")
            logger.info("Re-validating gaps after backfill...")

            report = basic_raw_checks(base_path, interval_ms)
            logger.info(f"Gaps after backfill: {report['gaps']}")
            logger.info(f"Missing candles after backfill: {report['missing_candles']}")
        else:
            logger.info("No gaps detected. Backfill skipped.")
    
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
    
    logger.info("Materializing Gold v1 layer...")
    
    con = duckdb.connect("warehouse.duckdb")
    
    try:
        # Drop previous table if exists
        con.execute("DROP TABLE IF EXISTS gold_v1")

        # Read SQL model
        gold_sql = Path("transformations/models/gold/gold_v1.sql").read_text()

        # Create materialized Gold table
        con.execute(f"CREATE TABLE gold_v1 AS {gold_sql}")

        # Basic validation
        row_count = con.execute("SELECT COUNT(*) FROM gold_v1").fetchone()[0]

        if row_count == 0:
            logger.error("Gold v1 table is empty.")
            raise Exception("Gold materialization failed")
        
        logger.info(f"Gold v1 rows: {row_count}")

        # Ensure storage path exists
        gold_storage_path = Path("storage/gold")
        gold_storage_path.mkdir(parents=True, exist_ok=True)

        file_path = gold_storage_path / "btcusdt_1h_v1.parquet"

        if file_path.exists():
            logger.error("Gold v1 parquet already exists.")
            raise Exception("Gold version is immutable")
        
        # Export to Parquet
        con.execute(f"COPY gold_v1 TO '{file_path}' (FORMAT PARQUET)")

        logger.info("Gold v1 layer materialized successfully.")
    
    finally:
        con.close()


def run_model_stage():
    logger.info("Running walk-forward experiment...")
    from models.model_stage import run_full_walkforward_experiment
    run_full_walkforward_experiment()

