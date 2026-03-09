import numpy as np
import pandas as pd

from models.walkforward.walkforward_runner import (
    prepare_features_and_target,
    generate_expanding_windows,
    prepare_walkforward_windows,
)
from models.robustness.robustness_runner import run_single_config
from tests.utils.mock_data import create_mock_dataset


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
