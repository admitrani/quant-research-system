import yaml
from pathlib import Path
from functools import lru_cache


CONFIG_PATH = Path(__file__).resolve().parent / "v1.yaml"

@lru_cache(maxsize=1)
def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    return config

def get_gold_path() -> Path:
    """Centralized resolver for the gold dataset parquet path."""
    config = load_config()
    symbol = config["market_data"]["symbols"][0].lower()
    return Path(__file__).resolve().parents[1] / "storage" / "gold" / f"{symbol}_1h_v1.parquet"
