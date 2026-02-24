import logging
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / "pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)

logger = logging.getLogger(__name__)


from orchestration.stages import (
    run_ingestion,
    run_silver_transformations,
    run_gold_transformations,
    run_model_stage
)

# Current execution mode: batch

def run_pipeline():

    logger.info("=" * 60)
    logger.info("Starting pipeline execution")

    try:
        logger.info("Stage 1: Ingestion")
        run_ingestion()

        logger.info("Stage 2: Silver transformations")
        run_silver_transformations()

        logger.info("Stage 3: Gold transformations")
        run_gold_transformations()

        logger.info("Stage 4: Model stage")
        run_model_stage()

        logger.info("Pipeline finished successfully.")

    except Exception as e:
        logger.exception("Pipeline failed with error.")
        raise

    finally:
        logger.info("=" * 60)

if __name__ == "__main__":
    run_pipeline()