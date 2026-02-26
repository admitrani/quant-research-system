import logging
from pathlib import Path
import argparse
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

def run_pipeline(start_stage=None, backfill_start=None):

    logger.info("=" * 60)
    logger.info("Starting pipeline execution")

    pipeline_start = time.time()

    stages = [
        ("ingestion", run_ingestion),
        ("silver", run_silver_transformations),
        ("gold", run_gold_transformations),
        ("model", run_model_stage)
    ]

    if start_stage:
        stage_names = [name for name, _ in stages]
        
        if start_stage not in stage_names:
            logger.error(f"Invalid stage specified: {start_stage}")
            raise ValueError(f"Invalid stage: {start_stage}")
        
        if backfill_start and start_stage and start_stage != "ingestion":
            logger.error("Backfill can only be used when starting from ingestion stage.")
            raise ValueError("Invalid backfill usage")
        
        start_index = stage_names.index(start_stage)
        stages = stages[start_index:]
    
    try:
        for stage_name, stage_func in stages:

            logger.info(f"Stage: {stage_name}")
            stage_start = time.time()

            if stage_name == "ingestion":
                stage_func(backfill_start)
            else:
                stage_func()

            duration = time.time() - stage_start
            logger.info(f"Stage {stage_name} completed in {duration:.2f}s")

        total_duration = time.time() - pipeline_start
        logger.info(f"Pipeline finished successfully in {total_duration:.2f}s.")

    except Exception:
        logger.exception("Pipeline failed.")
        raise

    finally:
        logger.info("=" * 60)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--stage", type=str, help="Start execution from specific stage")
    parser.add_argument("--backfill-start", type=str, help="Backfill start date (YYYY-MM-DD)")
    
    args = parser.parse_args()

    run_pipeline(start_stage=args.stage, backfill_start=args.backfill_start)