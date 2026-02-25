import logging
from pathlib import Path
import time

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

    pipeline_start = time.time()

    try:
        # Stage 1
        stage_start = time.time()
        logger.info("Stage 1: Ingestion")
        had_new_data = run_ingestion()
        logging.info(f"Stage 1 duration: {time.time() - stage_start:.2f}s")

        # Stage 2
        stage_start = time.time()
        logger.info("Stage 2: Silver transformations")
        run_silver_transformations(had_new_data)
        logging.info(f"Stage 2 duration: {time.time() - stage_start:.2f}s")

        # Stage 3
        stage_start = time.time()
        logger.info("Stage 3: Gold transformations")
        run_gold_transformations()
        logging.info(f"Stage 3 duration: {time.time() - stage_start:.2f}s")

        # Stage 4
        stage_start = time.time()
        logger.info("Stage 4: Model stage")
        run_model_stage()
        logging.info(f"Stage 4 duration: {time.time() - stage_start:.2f}s")

        logger.info("Pipeline finished successfully.")

    except Exception as e:
        logger.exception("Pipeline failed with error.")
        raise

    finally:
        total_duration = time.time() - pipeline_start
        logger.info(f"Total pipeline duration: {total_duration:.2f}s")
        logger.info("=" * 60)

if __name__ == "__main__":
    run_pipeline()