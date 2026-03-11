import logging
from pathlib import Path
import argparse
import time

from models.model_stage import main as run_walkforward
from models.model_selection.metrics_summary import main as run_metrics
from models.robustness.robustness_runner import main as run_robustness
from models.model_selection.candidate_selection import main as run_candidate
from models.utils import get_risk_policy, validate_risk_policy


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / "research.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)

logger = logging.getLogger(__name__)


def run_research(stage=None):

    logger.info("=" * 60)
    logger.info("Starting research pipeline")

    pipeline_start = time.time()

    stages = [
        ("walkforward", run_walkforward),
        ("metrics", run_metrics),
        ("robustness", run_robustness),
        ("candidate", run_candidate),
    ]

    risk = get_risk_policy()
    validate_risk_policy(risk)

    logger.info("Risk policy loaded:")
    for k, v in risk.items():
        logger.info(f"{k}: {v}")

    if stage and stage != "all":
        stage_names = [name for name, _ in stages]

        if stage not in stage_names:
            logger.error(f"Invalid stage specified: {stage}")
            raise ValueError(f"Invalid stage: {stage}")

        start_index = stage_names.index(stage)
        stages = stages[start_index:]

    try:

        for stage_name, stage_func in stages:

            logger.info(f"Stage: {stage_name}")

            stage_start = time.time()

            stage_func()

            duration = time.time() - stage_start
            logger.info(f"Stage {stage_name} completed in {duration:.2f}s")

        total_duration = time.time() - pipeline_start
        logger.info(f"Research pipeline finished successfully in {total_duration:.2f}s")

    except Exception:
        logger.exception("Research pipeline failed.")
        raise

    finally:
        logger.info("=" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        type=str,
        choices=["walkforward", "metrics", "robustness", "candidate", "all"],
        default="all",
        help="Start execution from research stage"
    )

    args = parser.parse_args()

    run_research(stage=args.stage)
