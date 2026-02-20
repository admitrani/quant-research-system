import logging
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / "pipeline.log"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Formatter común
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

# Handler consola
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Handler archivo
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)

# Evitar duplicados si reinicias
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


from orchestration.stages import (
    run_ingestion,
    run_silver_transformations,
    run_gold_transformations,
    run_model_stage
)

# Current execution mode: batch

def run_pipeline():

    logging.info("=" * 60)
    logging.info("Starting pipeline execution")

    try:
        logging.info("Stage 1: Ingestion")
        run_ingestion()

        logging.info("Stage 2: Silver transformations")
        run_silver_transformations()

        logging.info("Stage 3: Gold transformations")
        run_gold_transformations()

        logging.info("Stage 4: Model stage")
        run_model_stage()

        logging.info("Pipeline finished successfully.")

    except Exception as e:
        logging.exception("Pipeline failed with error.")
        raise

    finally:
        logging.info("=" * 60)

if __name__ == "__main__":
    run_pipeline()