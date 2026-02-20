from datetime import datetime
from ingestion.market_data import fetch_klines, save_raw, get_last_timestamp
from ingestion.audit import basic_raw_checks
from pathlib import Path
import logging


SYMBOL = "BTCUSDT"
INTERVAL = "1h"


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
        logging.info("Nothing to ingest.")
    else:
        save_raw(df, symbol=SYMBOL, interval=INTERVAL)
        logging.info(f"Ingested {len(df)} candles.")

    base_path = Path(
        f"storage/raw/source=binance/dataset=klines/"
        f"symbol={SYMBOL}/interval={INTERVAL}"
    )

    interval_ms = 3600000

    report = basic_raw_checks(base_path, interval_ms)
    
    logging.info("Raw ingestion summary:")
    logging.info(f"Total candles: {report['total_candles']}")
    logging.info(f"Duplicates: {report['duplicates']}")
    logging.info(f"Is sorted: {report['is_sorted']}")
    logging.info(f"Gaps detected: {report['gaps']}")
    logging.info(f"Missing candles: {report['missing_candles']}")

    if report["duplicates"] > 0:
        logging.warning("Duplicates detected in raw data.")

    if report["gaps"] > 0:
        logging.warning("Temporal gaps detected in raw data.")


def run_silver_transformations():
    pass


def run_gold_transformations():
    pass


def run_model_stage():
    pass

