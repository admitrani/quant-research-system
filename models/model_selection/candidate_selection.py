import pandas as pd
from pathlib import Path
import logging
import json

from models.utils import get_baseline_model_config

logger = logging.getLogger(__name__)

def load_results():

    project_root = Path(__file__).resolve().parents[2]
    results_path = project_root / "models" / "results"

    if not (results_path / "model_comparison_v1.csv").exists():
        raise FileNotFoundError("Run metrics stage first.")

    if not (results_path / "robustness_v1.csv").exists():
        raise FileNotFoundError("Run robustness stage first.")

    baseline_df = pd.read_csv(results_path / "model_comparison_v1.csv")
    robustness_df = pd.read_csv(results_path / "robustness_v1.csv")

    return baseline_df, robustness_df


def compute_robustness_summary(robustness_df):

    robustness_df = robustness_df[robustness_df["sharpe_global"].notna()]

    summary = []

    for model in robustness_df["model"].unique():

        df = robustness_df[robustness_df["model"] == model]

        summary.append({
            "model": model,
            "configs": len(df),
            "mean_sharpe": df["sharpe_global"].mean(),
            "std_sharpe": df["sharpe_global"].std(),
            "best_sharpe": df["sharpe_global"].max(),
            "negative_configs": (df["sharpe_global"] < 0).sum(),
        })

    return pd.DataFrame(summary)


def select_candidate_model(baseline_df, robustness_summary):

    merged = baseline_df.merge(robustness_summary, on="model", how="left")
    merged["selection_score"] = merged["mean_sharpe"] * 0.5 + merged["best_sharpe"] * 0.3 - merged["negative_configs"] * 0.1
    merged = merged.sort_values("selection_score", ascending=False)

    candidate = merged.iloc[0]

    return merged, candidate


def select_robust_config(robustness_df, model):

    if "final_equity_multiple" in robustness_df.columns:
        robustness_df = robustness_df[robustness_df["sharpe_global"].notna()]

    df = robustness_df[robustness_df["model"] == model].copy()

    # top 30% by Sharpe
    cutoff = df["sharpe_global"].quantile(0.7)
    robust_region = df[df["sharpe_global"] >= cutoff].copy()

    median_depth = robust_region["max_depth"].median()
    median_threshold = robust_region["threshold"].median()

    depth_range = robust_region["max_depth"].max() - robust_region["max_depth"].min()
    threshold_range = robust_region["threshold"].max() - robust_region["threshold"].min()

    depth_range = depth_range if depth_range > 0 else 1
    threshold_range = threshold_range if threshold_range > 0 else 1

    robust_region["distance"] = (
        ((robust_region["max_depth"] - median_depth) / depth_range).abs() +
        ((robust_region["threshold"] - median_threshold) / threshold_range).abs()
    )

    best_row = robust_region.sort_values("distance").iloc[0]

    selected_config = {
        "max_depth": int(best_row["max_depth"]),
        "threshold": float(best_row["threshold"])
    }
   
    return selected_config

def save_candidate(candidate, robust_config):

    project_root = Path(__file__).resolve().parents[2]
    results_path = project_root / "models" / "results"

    baseline_config = get_baseline_model_config(candidate["model"])
    threshold = baseline_config["threshold"]
    max_depth = baseline_config["max_depth"]
    robustness_df = pd.read_csv(results_path / "robustness_v1.csv")

    config_row = robustness_df[
        (robustness_df["model"] == candidate["model"]) &
        (robustness_df["max_depth"] == robust_config["max_depth"]) &
        (robustness_df["threshold"] == robust_config["threshold"])
    ].iloc[0]

    output_path = results_path / "candidate_v1.json"

    data = {
        "model": candidate["model"],

        "baseline_config": {
            "threshold": float(threshold),
            "max_depth": int(max_depth),
        },

        "baseline_metrics": {
            "sharpe": float(candidate["sharpe_global"]),
            "equity_multiple": float(candidate["final_equity_multiple"]),
        },

        "robust_config": {
            "threshold": float(robust_config["threshold"]),
            "max_depth": int(robust_config["max_depth"]),
        },

        "robust_config_metrics": {
            "sharpe_global": float(config_row["sharpe_global"]),
            "equity_multiple": float(config_row["final_equity_multiple"]),
            "mean_sharpe_window": float(config_row["mean_sharpe_window"]),
            "std_sharpe_window": float(config_row["std_sharpe_window"]),
            "negative_windows": int(config_row["negative_windows"]),
        },

        "robustness_summary": {
            "mean_sharpe": float(candidate["mean_sharpe"]),
            "best_sharpe": float(candidate["best_sharpe"]),
            "configs_tested": int(candidate["configs"]),
        }
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"Artifact produced: {output_path}")


def main():

    baseline_df, robustness_df = load_results()
    robustness_summary = compute_robustness_summary(robustness_df)
    merged, candidate = select_candidate_model(baseline_df, robustness_summary)
    robust_config = select_robust_config(robustness_df, candidate["model"])
    
    logger.info("MODEL SCORECARD")
    logger.info("\n%s", merged.to_string())
    logger.info("SELECTED MODEL")
    logger.info("\n%s", candidate.to_string())
    logger.info(f"ROBUST CONFIGURATION: {robust_config}")

    save_candidate(candidate, robust_config)


if __name__ == "__main__":
    main()
