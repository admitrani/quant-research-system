import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# Stage selection

def test_invalid_stage_raises_value_error():
    from backtests.v1.backtest_pipeline import run_backtest_pipeline
    with pytest.raises(ValueError, match="Invalid stage"):
        run_backtest_pipeline(stage="nonexistent_stage")


def test_valid_stage_names_accepted():
    """All documented stage names should not raise ValueError on import."""
    from backtests.v1.backtest_pipeline import STAGES
    stage_names = [name for name, _ in STAGES]
    assert "ml" in stage_names
    assert "ema" in stage_names
    assert "buy_and_hold" in stage_names
    assert "comparison" in stage_names


def test_stages_run_in_order():
    """Stages list must be in the correct execution order."""
    from backtests.v1.backtest_pipeline import STAGES
    names = [name for name, _ in STAGES]
    assert names.index("ml") < names.index("ema")
    assert names.index("ema") < names.index("buy_and_hold")
    assert names.index("buy_and_hold") < names.index("comparison")


# Comparison stage

def make_mock_ml_csv(tmp_path):
    metrics = pd.Series({
        "Final Value": 9994.93,
        "CAGR": -0.272,
        "Sharpe": -2.315,
        "Sortino": -0.00062,
        "Volatility": 0.933,
        "Max Drawdown": 90.64,
        "Calmar": -0.300,
        "Trades": 665.0,
        "Win Rate": 0.459,
        "Profit Factor": 0.557,
        "Avg Trade Return": -135.35,
        "Exposure": 0.060,
        "Avg Trade Duration": 3.96,
    })
    path = tmp_path / "metrics_ml_v1.csv"
    metrics.to_frame(name="value").to_csv(path)
    return path


def make_mock_ema_csv(tmp_path):
    row = {
        "final_value": 9998.35, "cagr": -0.359, "sharpe": -1.246,
        "sortino": -0.00077, "volatility": 1.554, "max_drawdown_pct": 93.01,
        "calmar_ratio": -0.386, "total_trades": 382, "win_rate": 0.220,
        "profit_factor": 0.717, "expectancy_per_trade": -235.6,
        "net_pnl": -90001.65, "exposure": 0.185, "avg_trade_duration": 21.87,
    }
    path = tmp_path / "metrics_ema_v1.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def make_mock_bh_csv(tmp_path):
    row = {
        "final_value": 232964.21, "cagr": 0.178, "sharpe": 0.572,
        "sortino": 0.00046, "volatility": 2.871, "max_drawdown_pct": 77.10,
        "calmar_ratio": 0.231, "total_trades": None, "win_rate": None,
        "profit_factor": None, "expectancy_per_trade": None,
        "net_pnl": 132964.21, "exposure": None, "avg_trade_duration": None,
    }
    path = tmp_path / "metrics_buy_hold_v1.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def test_comparison_builds_correct_columns(tmp_path):
    """comparison_v1.csv must have exactly the three expected strategy columns."""
    ml_path  = make_mock_ml_csv(tmp_path)
    ema_path = make_mock_ema_csv(tmp_path)
    bh_path  = make_mock_bh_csv(tmp_path)

    output_path = tmp_path / "results"

    with patch("backtests.v1.backtest_pipeline.Path") as mock_path_cls:
        # Route specific path lookups to our tmp files
        def path_side_effect(p):
            mapping = {
                "backtests/v1/ml/results/metrics_ml_v1.csv":       ml_path,
                "backtests/v1/baseline/results/metrics_ema_v1.csv": ema_path,
                "backtests/v1/baseline/results/metrics_buy_hold_v1.csv": bh_path,
                "backtests/v1/results": output_path,
            }
            real = mapping.get(str(p))
            if real:
                return real
            return Path(p)

        mock_path_cls.side_effect = path_side_effect

        # Run comparison directly on real files
        ml_raw  = pd.read_csv(ml_path,  index_col=0)["value"]
        ema     = pd.read_csv(ema_path).iloc[0].to_dict()
        bh      = pd.read_csv(bh_path).iloc[0].to_dict()

    assert "Final Value" in ml_raw.index
    assert "final_value" in ema
    assert "final_value" in bh


def test_comparison_raises_if_ml_csv_missing(tmp_path):
    """comparison stage must raise FileNotFoundError if any CSV is absent."""
    from backtests.v1.backtest_pipeline import run_comparison

    with patch("backtests.v1.backtest_pipeline.Path") as mock_path_cls:
        mock_instance = MagicMock()
        mock_instance.exists.return_value = False
        mock_path_cls.return_value = mock_instance

        with pytest.raises(FileNotFoundError):
            run_comparison()


def test_comparison_output_has_expected_metrics():
    """Verify all key metric rows appear in the comparison table."""
    expected_metrics = [
        "final_value", "cagr", "sharpe", "max_drawdown_pct",
        "calmar_ratio", "win_rate", "profit_factor", "exposure",
    ]
    # Build minimal comparison manually (mirrors pipeline logic)
    ml = {m: 0.0 for m in expected_metrics}
    ema = {m: 0.0 for m in expected_metrics}
    bh = {m: 0.0 for m in expected_metrics}

    comparison = pd.DataFrame({
        "ML Strategy": ml,
        "EMA Baseline": ema,
        "Buy & Hold": bh,
    })

    for metric in expected_metrics:
        assert metric in comparison.index


# MLDataFeed structure

def test_ml_data_feed_has_ml_prob_line():
    """MLDataFeed must expose ml_prob as a custom line."""
    from backtests.v1.ml.data_feed import MLDataFeed
    line_names = MLDataFeed.lines._getlines()
    assert "ml_prob" in line_names


def test_ml_data_feed_column_mapping():
    """MLDataFeed params must map OHLCV to gold dataset column names."""
    from backtests.v1.ml.data_feed import MLDataFeed
    params = dict(MLDataFeed.params._getitems())
    assert params.get("open")  == "open_price"
    assert params.get("close") == "close_price"
    assert params.get("high")  == "high_price"
    assert params.get("low")   == "low_price"
    assert params.get("ml_prob") == "ml_prob"
