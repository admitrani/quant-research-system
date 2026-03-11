import pandas as pd
import numpy as np
import shutil
from pathlib import Path
import logging
from scipy.stats import linregress
from models.walkforward.walkforward_runner import (load_gold_dataset, prepare_features_and_target, prepare_walkforward_windows, generate_expanding_windows, run_walkforward_for_model)
from config.config_loader import load_config
from models.utils import get_annualization_factor
from models.metrics import compute_daily_sharpe


logger = logging.getLogger(__name__)


def run_full_walkforward_experiment():

    config = load_config()
    annualization_factor = get_annualization_factor()
    threshold = config["system"]["modeling"]["threshold"]
    initial_train_years = config["system"]["validation"]["initial_train_years"]
    test_months = config["system"]["validation"]["test_months"]
    models = config["system"]["modeling"]["models"]

    logger.info("Loading gold dataset...")
    df = load_gold_dataset()
    logger.info(f"Total rows: {len(df)}")

    X, y = prepare_features_and_target(df)
    windows = generate_expanding_windows(df, initial_train_years=initial_train_years, test_months=test_months)
    logger.info(f"Total windows generated: {len(windows)}")

    summary_rows = []
    prepared_windows = prepare_walkforward_windows(X, y, windows, df)

    for model_name in models:

        logger.info(f"Running walk-forward for model: {model_name}")

        results_df, equity_curve, returns, oos_dates = run_walkforward_for_model(
            prepared_windows,
            model_name=model_name,
            threshold=threshold,
            annualization_factor=annualization_factor,
            save_results=True,
        )

        sharpe_global = compute_daily_sharpe(returns, oos_dates)

        slope_auc = linregress(results_df["window"], results_df["auc"]).slope
        slope_sharpe = linregress(results_df["window"], results_df["sharpe"]).slope

        logger.info(
            f"{model_name} | mean_auc={results_df['auc'].mean():.4f} "
            f"| sharpe_global={sharpe_global:.4f}"
        )

        if np.isnan(equity_curve).any():
            logger.warning(f"{model_name}: NaNs detected in equity curve.")

        if equity_curve[-1] <= 0:
            logger.warning(f"{model_name}: Final equity <= 0.")
        
        neg_windows = (results_df["sharpe"] < 0).sum()
        logger.info(f"{model_name} | negative_windows={neg_windows}/{len(results_df)}")

        if len(returns) == 0:
            logger.error(f"{model_name}: No OOS returns generated.")
            raise ValueError("No OOS returns generated.")

        if np.all(returns == 0):
            logger.warning(f"{model_name}: All OOS returns are zero.")

        summary_rows.append({
            "model": model_name,
            "mean_auc": results_df["auc"].mean(),
            "std_auc": results_df["auc"].std(),
            "mean_accuracy": results_df["accuracy"].mean(),
            "std_accuracy": results_df["accuracy"].std(),
            "mean_sharpe_window": results_df["sharpe"].mean(),
            "std_sharpe_window": results_df["sharpe"].std(),
            "sharpe_global_oos": sharpe_global,
            "windows_auc_below_random": (results_df["auc"] < 0.5).sum(),
            "windows_sharpe_negative": (results_df["sharpe"] < 0).sum(),
            "auc_slope": slope_auc,
            "sharpe_slope": slope_sharpe,
            "final_equity_multiple": equity_curve[-1],
        })

    summary_df = pd.DataFrame(summary_rows)

    project_root = Path(__file__).resolve().parent
    results_path = project_root / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(results_path / "walkforward_summary_v1.csv", index=False)
    shutil.copy("config/v1.yaml", results_path / "config_snapshot_v1.yaml")

    print("Walk-forward experiment completed.")


def main():
    run_full_walkforward_experiment()


if __name__ == "__main__":
    main()
