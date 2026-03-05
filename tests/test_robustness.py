import numpy as np
import pandas as pd

from models.walkforward.walkforward_runner import (
    prepare_features_and_target,
    generate_expanding_windows,
    prepare_walkforward_windows,
)
from models.robustness.robustness_runner import run_single_config


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


def test_robustness_single_config_output():

    df = create_mock_dataset()
    X, y = prepare_features_and_target(df)
    windows = generate_expanding_windows(df, initial_train_years=3, test_months=6)

    prepared_windows = prepare_walkforward_windows(X, y, windows, df)

    result = run_single_config(
        model_name="rf",
        depth=4,
        threshold=0.5,
        prepared_windows=prepared_windows,
        df=df,
        annualization_factor=365 * 24
    )

    assert "model" in result
    assert "sharpe_global" in result
    assert "final_equity_multiple" in result
