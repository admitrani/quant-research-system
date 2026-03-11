import pandas as pd
import numpy as np
import logging
from pathlib import Path

from models.walkforward.walkforward_runner import load_gold_dataset, prepare_features_and_target, generate_expanding_windows, prepare_walkforward_windows, run_walkforward_for_model
from models.metrics import compute_daily_sharpe
from models.utils import get_annualization_factor, get_baseline_model_config
from config.config_loader import load_config

logger = logging.getLogger(__name__)

def run_baseline_walkforward(model_name, prepared_windows, annualization_factor, threshold, max_depth=None):

    wf_results, equity_curve, returns, oos_dates = run_walkforward_for_model(
        prepared_windows=prepared_windows,
        model_name=model_name,
        threshold=threshold,
        max_depth=max_depth,
        annualization_factor=annualization_factor,
        save_results=False,
    )

    sharpe_global = compute_daily_sharpe(returns, oos_dates)

    metrics = {
        "model": model_name,
        "sharpe_global": sharpe_global,
        "mean_sharpe_window": wf_results["sharpe"].mean(),
        "std_sharpe_window": wf_results["sharpe"].std(),
        "negative_windows": (wf_results["sharpe"] < 0).sum(),
        "expectancy_mean": wf_results["expectancy"].mean(),
        "final_equity_multiple": equity_curve[-1],
    }

    return metrics


def main():

    config = load_config()
    annualization_factor = get_annualization_factor()

    logger.info("Experiment configuration")
    logger.info(f"Train years: {config['system']['validation']['initial_train_years']}")
    logger.info(f"Test months: {config['system']['validation']['test_months']}")
    logger.info(f"Models evaluated: {config['system']['modeling']['models']}")
    logger.info(f"Baseline threshold: {config['system']['modeling']['threshold']}")
    logger.info(f"Annualization factor: {annualization_factor}")

    initial_train_years = config["system"]["validation"]["initial_train_years"]
    test_months = config["system"]["validation"]["test_months"]
    
    logger.info("Loading gold dataset...")
    df = load_gold_dataset()

    logger.info(f"Dataset loaded: {len(df)} rows")

    logger.info("Preparing features and target...")
    X, y = prepare_features_and_target(df)
    logger.info(f"Number of features: {len(X.columns)}")

    logger.info("Generating walk-forward windows...")
    windows = generate_expanding_windows(df, initial_train_years=initial_train_years, test_months=test_months,)
    logger.info(f"Walk-forward windows generated: {len(windows)}")

    logger.info("Preparing scaled windows...")
    prepared_windows = prepare_walkforward_windows(X, y, windows, df)

    results = []

    models = config["system"]["modeling"]["models"]

    for model_name in models:

        logger.info(f"Running baseline walk-forward for {model_name}")

        baseline_config = get_baseline_model_config(model_name)
        threshold = baseline_config["threshold"]
        max_depth = baseline_config["max_depth"]

        logger.info(f"Baseline config → threshold={threshold}, max_depth={max_depth}")

        metrics = run_baseline_walkforward(
            model_name=model_name,
            prepared_windows=prepared_windows,
            annualization_factor=annualization_factor,
            threshold=threshold,
            max_depth=max_depth,
        )

        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("sharpe_global", ascending=False)

    print("\nMODEL COMPARISON\n")
    logger.info("Model comparison results:\n%s", results_df.to_string())

    project_root = Path(__file__).resolve().parents[2]
    results_path = project_root / "models" / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    output_file = results_path / "model_comparison_v1.csv"
    results_df.to_csv(output_file, index=False)

    logger.info(f"Artifact produced: {output_file}")


if __name__ == "__main__":
    main()


