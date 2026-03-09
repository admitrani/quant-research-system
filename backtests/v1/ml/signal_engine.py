import pandas as pd
from models.walkforward.walkforward_runner import (
    load_gold_dataset,
    prepare_features_and_target,
    generate_expanding_windows,
    split_window,
    scale_window,
)
from models.model_factory import get_model
from models.candidate_loader import get_candidate_model_config
from config.config_loader import load_config
import logging

logger = logging.getLogger(__name__)


def generate_ml_probabilities():
    """
    Run walk-forward signal generation using the candidate model config.

    Returns
    -------
    df_probs : pd.DataFrame
        Gold dataset with an added `ml_prob` column for OOS bars only.
    candidate : dict
        Candidate config dict (model, threshold, max_depth) — returned
        so the caller does not need to reload it.
    """

    config = load_config()
    candidate = get_candidate_model_config()

    model_name = candidate["model"]
    max_depth = candidate["max_depth"]
    initial_train_years = config["system"]["validation"]["initial_train_years"]
    test_months = config["system"]["validation"]["test_months"]

    logger.info(f"Generating ML probabilities | model: {model_name} | max_depth: {max_depth}")
    logger.info(f"Walk-forward config — initial_train_years: {initial_train_years} | test_months: {test_months}")

    df = load_gold_dataset()
    X, y = prepare_features_and_target(df)
    windows = generate_expanding_windows(
        df,
        initial_train_years=initial_train_years,
        test_months=test_months,
    )

    logger.info(f"Walk-forward windows: {len(windows)}")

    prob_series = []

    for i, window in enumerate(windows):
        X_train, y_train, X_test, y_test = split_window(X, y, window)
        X_train_s, X_test_s, _ = scale_window(X_train, X_test)

        model = get_model(model_name, max_depth=max_depth)
        model.fit(X_train_s, y_train)

        y_proba = model.predict_proba(X_test_s)[:, 1]
        prob_series.append(pd.Series(y_proba, index=X_test.index))

        logger.info(f"Window {i + 1}/{len(windows)} | train: {X_train.index[0].date()} → {X_train.index[-1].date()} | test: {X_test.index[0].date()} → {X_test.index[-1].date()}")

    all_probs = pd.concat(prob_series)

    df_probs = df.copy()
    df_probs["ml_prob"] = all_probs
    df_probs = df_probs.dropna(subset=["ml_prob"])

    logger.info(f"OOS bars with ML signal: {len(df_probs)}")

    return df_probs, candidate
