import pandas as pd
import logging
from pathlib import Path
from joblib import Parallel, delayed

from models.walkforward.walkforward_runner import load_gold_dataset, prepare_features_and_target, generate_expanding_windows, prepare_walkforward_windows, run_walkforward_for_model
from config.config_loader import load_config
from models.utils import get_annualization_factor
from models.metrics import compute_global_sharpe


logger = logging.getLogger(__name__)


def run_single_config(
    model_name,
    depth,
    threshold,
    prepared_windows,
    df,
    annualization_factor
):

    wf_results, equity_curve, returns = run_walkforward_for_model(
        prepared_windows,
        model_name=model_name,
        threshold=threshold,
        max_depth=depth,
        annualization_factor=annualization_factor,
        save_results=False,
    )

    sharpe_global = compute_global_sharpe(
        returns,
        annualization_factor
    )

    return {
        "model": model_name,
        "max_depth": depth,
        "threshold": threshold,
        "sharpe_global": sharpe_global,
        "final_equity_multiple": equity_curve[-1],
        "mean_sharpe_window": wf_results["sharpe"].mean(),
        "std_sharpe_window": wf_results["sharpe"].std(),
        "negative_windows": (wf_results["sharpe"] < 0).sum(),
    }


def run_robustness_experiment():

    config = load_config()
    annualization_factor = get_annualization_factor()

    initial_train_years = config["system"]["validation"]["initial_train_years"]
    test_months = config["system"]["validation"]["test_months"]

    threshold_grid = [0.45, 0.5, 0.55, 0.6]

    rf_depths = [4, 6, 8]
    xgb_depths = [3, 4, 5]

    df = load_gold_dataset()
    X, y = prepare_features_and_target(df)
    windows = generate_expanding_windows(df, initial_train_years=initial_train_years, test_months=test_months)
    prepared_windows = prepare_walkforward_windows(X, y, windows, df)

    jobs = []

    for model_name in ["rf", "xgb"]:
        
        logger.info(f"Preparing robustness jobs for {model_name}")

        depths = rf_depths if model_name == "rf" else xgb_depths

        for depth in depths:
            for threshold in threshold_grid:
                
                jobs.append(
                    delayed(run_single_config)(
                        model_name,
                        depth,
                        threshold,
                        prepared_windows,
                        df,
                        annualization_factor
                    )
                )

    logger.info(f"Running {len(jobs)} robustness configurations in parallel")
    results = Parallel(n_jobs=-1, backend="loky")(jobs)

    results_df = pd.DataFrame(results)

    top_configs = results_df.sort_values("sharpe_global", ascending=False).head(3)
    logger.info("Top 3 robustness configurations:")
    logger.info("\n%s", top_configs.to_string())

    project_root = Path(__file__).resolve().parents[2]
    results_path = project_root / "models" / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(results_path / "robustness_v1.csv", index=False)
    print("Robustness experiment completed.")