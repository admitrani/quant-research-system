import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from tests.utils.mock_data import create_mock_dataset
from models.walkforward.walkforward_runner import (
    prepare_features_and_target,
    generate_expanding_windows,
    split_window,
    scale_window,
)
from models.model_factory import get_model


# Helpers

def generate_mock_probabilities(df, model_name="xgb", max_depth=4,
                                 initial_train_years=2, test_months=6):
    """
    Replicate signal_engine.generate_ml_probabilities() using mock data
    so tests run without the real gold parquet.
    """
    X, y = prepare_features_and_target(df)
    windows = generate_expanding_windows(
        df, initial_train_years=initial_train_years, test_months=test_months
    )

    prob_series = []
    for window in windows:
        X_train, y_train, X_test, _ = split_window(X, y, window)
        X_train_s, X_test_s, _ = scale_window(X_train, X_test)
        model = get_model(model_name, max_depth=max_depth)
        model.fit(X_train_s, y_train)
        probs = model.predict_proba(X_test_s)[:, 1]
        prob_series.append(pd.Series(probs, index=X_test.index))

    all_probs = pd.concat(prob_series)
    df_out = df.copy()
    df_out["ml_prob"] = all_probs
    df_out = df_out.dropna(subset=["ml_prob"])
    return df_out


# Output structure

def test_signal_engine_output_has_ml_prob_column():
    df = create_mock_dataset()
    result = generate_mock_probabilities(df)
    assert "ml_prob" in result.columns


def test_signal_engine_no_nan_in_ml_prob():
    df = create_mock_dataset()
    result = generate_mock_probabilities(df)
    assert result["ml_prob"].isna().sum() == 0


def test_signal_engine_probabilities_in_0_1_range():
    df = create_mock_dataset()
    result = generate_mock_probabilities(df)
    assert result["ml_prob"].between(0.0, 1.0).all()


def test_signal_engine_output_length_matches_oos_windows():
    """Total OOS bars should equal sum of test window lengths."""
    df = create_mock_dataset()
    windows = generate_expanding_windows(df, initial_train_years=2, test_months=6)
    expected_oos = sum(
        ((df.index >= w["test_start"]) & (df.index < w["test_end"])).sum()
        for w in windows
    )
    result = generate_mock_probabilities(df)
    assert len(result) == expected_oos


# No lookahead bias

def test_signal_engine_future_return_not_in_features():
    """future_return must never appear in X used for training."""
    df = create_mock_dataset()
    X, y = prepare_features_and_target(df)
    assert "future_return" not in X.columns
    assert "label" not in X.columns


def test_signal_engine_train_strictly_before_test():
    """For every window, all train indices must be strictly before test indices."""
    df = create_mock_dataset()
    X, y = prepare_features_and_target(df)
    windows = generate_expanding_windows(df, initial_train_years=2, test_months=6)
    for window in windows:
        X_train, _, X_test, _ = split_window(X, y, window)
        assert X_train.index.max() < X_test.index.min()


def test_signal_engine_oos_windows_are_contiguous():
    """Test windows must be contiguous — no gap between consecutive windows."""
    df = create_mock_dataset()
    windows = generate_expanding_windows(df, initial_train_years=2, test_months=6)
    for i in range(1, len(windows)):
        assert windows[i]["test_start"] == windows[i - 1]["test_end"]


def test_signal_engine_train_grows_each_window():
    """In expanding walk-forward, each train set must be larger than the previous."""
    df = create_mock_dataset()
    X, y = prepare_features_and_target(df)
    windows = generate_expanding_windows(df, initial_train_years=2, test_months=6)
    sizes = []
    for window in windows:
        X_train, _, _, _ = split_window(X, y, window)
        sizes.append(len(X_train))
    assert sizes == sorted(sizes), "Train set must grow monotonically"


# Candidate config passthrough

def test_signal_engine_returns_candidate_alongside_df():
    """
    signal_engine.generate_ml_probabilities() must return (df, candidate)
    so run_backtest.py does not need a second call to get_candidate_model_config.
    """
    mock_candidate = {"model": "xgb", "threshold": 0.6, "max_depth": 4}
    mock_df = create_mock_dataset()
    mock_df["ml_prob"] = 0.5

    with patch(
        "backtests.v1.ml.signal_engine.get_candidate_model_config",
        return_value=mock_candidate,
    ):
        with patch(
            "backtests.v1.ml.signal_engine.load_gold_dataset",
            return_value=mock_df,
        ):
            from backtests.v1.ml.signal_engine import generate_ml_probabilities
            result = generate_ml_probabilities()

    assert isinstance(result, tuple)
    assert len(result) == 2
    df_out, candidate = result
    assert "ml_prob" in df_out.columns
    assert candidate["model"] == "xgb"
    