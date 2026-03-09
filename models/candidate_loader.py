import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_candidate():

    project_root = Path(__file__).resolve().parents[1]
    candidate_path = project_root / "models" / "results" / "candidate_v1.json"

    with open(candidate_path) as f:
        candidate = json.load(f)

    return candidate


def get_candidate_model_config():

    candidate = load_candidate()

    robust = candidate.get("robust_config")
    baseline = candidate.get("baseline_config")

    config = robust if robust else baseline
    config_type = "robust_config" if robust else "baseline_config"

    logger.info("Loading candidate model configuration")
    logger.info(f"Model: {candidate['model']}")
    logger.info(f"Config source: {config_type}")
    logger.info(f"Threshold: {config['threshold']}")
    logger.info(f"Max depth: {config['max_depth']}")

    return {
        "model": candidate["model"],
        "threshold": config["threshold"],
        "max_depth": config["max_depth"],
    }
