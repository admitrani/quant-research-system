import pandas as pd
import numpy as np

from models.walkforward.walkforward_runner import (
    prepare_features_and_target,
    generate_expanding_windows,
    prepare_walkforward_windows,
    split_window,
    scale_window,
    run_walkforward_for_model,
)

from models.metrics import compute_global_sharpe, compute_daily_sharpe
from tests.utils.mock_data import create_mock_dataset


# Window generation test

def test_generate_expanding_windows():

    df = create_mock_dataset()
    windows = generate_expanding_windows(df, initial_train_years=2, test_months=6)

    assert len(windows) > 0
    assert windows[0]["train_start"] < windows[0]["train_end"]
    assert windows[0]["train_end"] == windows[0]["test_start"]


# Split test (no leakage)

def test_split_no_overlap():

    df = create_mock_dataset()
    X, y = prepare_features_and_target(df)
    windows = generate_expanding_windows(df, initial_train_years=2, test_months=6)

    X_train, y_train, X_test, y_test = split_window(X, y, windows[0])

    assert X_train.index.max() < X_test.index.min()
    assert len(X_train) > 0
    assert len(X_test) > 0


# Scaling test (no train leakage)

def test_scaling_no_data_leakage():

    df = create_mock_dataset()
    X, y = prepare_features_and_target(df)
    windows = generate_expanding_windows(df, initial_train_years=2, test_months=6)

    X_train, y_train, X_test, y_test = split_window(X, y, windows[0])
    X_train_s, X_test_s, scaler = scale_window(X_train, X_test)

    # Shape preserved
    assert X_train_s.shape == X_train.shape
    assert X_test_s.shape == X_test.shape

    # Verify scaler was fit ONLY on training data
    assert np.allclose(scaler.mean_, X_train.mean(axis=0), rtol=1e-3)

    # Verify test set was transformed using training statistics
    # (if scaler had been fit on test data too, this would fail)
    X_test_manually = (X_test.values - scaler.mean_) / scaler.scale_
    assert np.allclose(X_test_s, X_test_manually, rtol=1e-5)


# Caching test

def test_prepare_walkforward_windows_structure():

    df = create_mock_dataset()
    X, y = prepare_features_and_target(df)
    windows = generate_expanding_windows(df, initial_train_years=2, test_months=6)

    prepared_windows = prepare_walkforward_windows(X, y, windows, df)

    assert len(prepared_windows) == len(windows)

    for w in prepared_windows:

        assert "X_train" in w
        assert "y_train" in w
        assert "X_test" in w
        assert "y_test" in w
        assert "future_returns" in w

        assert len(w["X_test"]) == len(w["future_returns"])


# Walkforward structural integrity

def test_walkforward_structural_integrity():

    df = create_mock_dataset()
    X, y = prepare_features_and_target(df)
    windows = generate_expanding_windows(df, initial_train_years=2, test_months=6)

    prepared_windows = prepare_walkforward_windows(X, y, windows, df)

    results_df, equity_curve, returns, oos_dates = run_walkforward_for_model(
        prepared_windows,
        model_name="rf",
        threshold=0.5,
        annualization_factor=365 * 24,
        save_results=False,
    )

    # OOS windows contiguous
    prev_end = None
    for w in windows:
        if prev_end is not None:
            assert w["test_start"] == prev_end
        prev_end = w["test_end"]

    # Expected OOS length
    total_expected = 0

    for w in windows:
        mask = (df.index >= w["test_start"]) & (df.index < w["test_end"])
        total_expected += mask.sum()

    assert total_expected == len(returns)

    # Equity consistency
    calc_equity = np.prod(1 + returns)
    assert equity_curve[-1] > 0

    assert np.isclose(calc_equity, equity_curve[-1])

    # No NaNs
    assert not np.isnan(returns).any()
    assert not np.isnan(equity_curve).any()

# Sharpe function test
def test_compute_global_sharpe_basic():

    returns = np.array([0.01, -0.01, 0.02, -0.02])

    sharpe = compute_global_sharpe(returns, annualization_factor=365)

    assert isinstance(sharpe, float)

def test_compute_daily_sharpe_basic():
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", periods=200, freq="h")
    returns = np.random.normal(0.0001, 0.01, 200)
    
    sharpe = compute_daily_sharpe(returns, dates)
    
    assert isinstance(sharpe, float)
    assert -15 < sharpe < 15

def test_compute_daily_sharpe_zero_std():
    dates = pd.date_range("2021-01-01", periods=48, freq="h")
    returns = np.zeros(48)
    
    sharpe = compute_daily_sharpe(returns, dates)
    
    assert np.isnan(sharpe)
