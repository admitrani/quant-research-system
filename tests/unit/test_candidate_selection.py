import pandas as pd
import numpy as np

from models.model_selection.candidate_selection import select_candidate_model, select_robust_config

def test_candidate_selection_picks_best_model():

    baseline_df = pd.DataFrame({
        "model": ["logistic", "rf", "xgb"],
        "sharpe_global": [0.5, 1.0, 0.9],
        "mean_sharpe_window": [0.6, 1.2, 1.1],
        "std_sharpe_window": [1.0, 0.8, 0.9],
        "negative_windows": [4, 2, 3],
        "expectancy_mean": [0.00002, 0.00005, 0.00004],
        "final_equity_multiple": [1.5, 3.0, 2.5],
    })

    robustness_summary = pd.DataFrame({
        "model": ["logistic", "rf", "xgb"],
        "configs": [0, 12, 12],
        "mean_sharpe": [np.nan, 0.8, 0.7],
        "std_sharpe": [np.nan, 0.2, 0.3],
        "best_sharpe": [np.nan, 1.3, 1.1],
        "negative_configs": [np.nan, 1, 2],
    })

    merged, candidate = select_candidate_model(baseline_df, robustness_summary)

    assert candidate["model"] == "rf"


def test_select_robust_config_returns_valid_config():

    robustness_df = pd.DataFrame({
        "model": ["rf"] * 4,
        "max_depth": [4, 6, 8, 6],
        "threshold": [0.5, 0.55, 0.6, 0.55],
        "sharpe_global": [0.8, 1.2, 0.9, 1.1]
    })

    config = select_robust_config(robustness_df, "rf")

    assert "max_depth" in config
    assert "threshold" in config