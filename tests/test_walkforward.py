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

from models.metrics import compute_global_sharpe

# Helper: synthetic dataset (no dependency on BTC)

def create_mock_dataset():

    dates = pd.date_range("2019-01-01", periods=24*365*5, freq="h")

    df = pd.DataFrame({
        "open_time_utc": dates,
        "return_1h": np.random.normal(0, 0.01, len(dates)),
        "return_3h": np.random.normal(0, 0.01, len(dates)),
        "return_12h": np.random.normal(0, 0.01, len(dates)),
        "volatility_12h": np.random.normal(0.01, 0.005, len(dates)),
        "ma20_distance": np.random.normal(0, 0.02, len(dates)),
        "volume_zscore": np.random.normal(0, 1, len(dates)),
        "future_return": np.random.normal(0, 0.01, len(dates)),
        "label": np.random.randint(0, 2, len(dates)),
    })

    df.set_index("open_time_utc", inplace=True)
    return df


# Window generation test

def test_generate_expanding_windows():

    df = create_mock_dataset()
    windows = generate_expanding_windows(df, initial_train_years=3, test_months=6)

    assert len(windows) > 0
    assert windows[0]["train_start"] < windows[0]["train_end"]
    assert windows[0]["train_end"] == windows[0]["test_start"]


# Split test (no leakage)

def test_split_no_overlap():

    df = create_mock_dataset()
    X, y = prepare_features_and_target(df)
    windows = generate_expanding_windows(df, initial_train_years=3, test_months=6)

    X_train, y_train, X_test, y_test = split_window(X, y, windows[0])

    assert X_train.index.max() < X_test.index.min()
    assert len(X_train) > 0
    assert len(X_test) > 0


# Scaling test (no train leakage)

def test_scaling_no_data_leakage():

    df = create_mock_dataset()
    X, y = prepare_features_and_target(df)
    windows = generate_expanding_windows(df, initial_train_years=3, test_months=6)

    X_train, y_train, X_test, y_test = split_window(X, y, windows[0])
    X_train_s, X_test_s, scaler = scale_window(X_train, X_test)

    # Train should have mean approx 0 after scaling
    assert np.allclose(X_train_s.mean(axis=0), 0, atol=1e-1)
    assert X_train_s.shape == X_train.shape
    assert X_test_s.shape == X_test.shape


# Caching test

def test_prepare_walkforward_windows_structure():

    df = create_mock_dataset()
    X, y = prepare_features_and_target(df)
    windows = generate_expanding_windows(df, initial_train_years=3, test_months=6)

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
    windows = generate_expanding_windows(df, initial_train_years=3, test_months=6)

    prepared_windows = prepare_walkforward_windows(X, y, windows, df)

    results_df, equity_curve, returns = run_walkforward_for_model(
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