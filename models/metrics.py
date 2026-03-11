import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression
from models.utils import get_annualization_factor


def compute_classification_metrics(y_true, y_proba, threshold=0.5):

    y_pred = (y_proba > threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "positive_rate_pred": y_pred.mean(),
        "positive_rate_true": y_true.mean(),
    }

    return metrics

def compute_global_sharpe(returns, annualization_factor):
    """Compute annualized Sharpe ratio. Returns np.nan if no trades (std=0)."""
    mean_ret = returns.mean()
    std_ret = returns.std()

    if std_ret == 0:
        return np.nan

    return (mean_ret / std_ret) * np.sqrt(annualization_factor)

def compute_daily_sharpe(strategy_returns, test_dates):
    """
    Compute Sharpe using daily aggregated returns × sqrt(365).
    Matches the calculation used in the backtest pipeline.
    """
    series = pd.Series(strategy_returns, index=test_dates)
    daily = series.resample("D").apply(lambda x: (1 + x).prod() - 1).dropna()
    
    if len(daily) < 2:
        return np.nan
    
    mean_d = daily.mean()
    std_d = daily.std()
    
    if std_d == 0:
        return np.nan
    
    return (mean_d / std_d) * np.sqrt(365)

def compute_vectorized_trading_metrics(future_returns, y_proba, threshold, annualization_factor):

    signal = (y_proba > threshold).astype(int)

    strategy_returns = signal * future_returns

    mean_ret = strategy_returns.mean()
    std_ret = strategy_returns.std()

    sharpe = compute_global_sharpe(strategy_returns, annualization_factor)

    hit_ratio = (strategy_returns > 0).mean()

    expectancy = mean_ret

    return {
        "mean_return": mean_ret,
        "std_return": std_ret,
        "sharpe": sharpe,
        "hit_ratio": hit_ratio,
        "expectancy": expectancy,
    }

def compute_degradation_slope(values):

    x = np.arange(len(values)).reshape(-1, 1)
    y = np.array(values).reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)

    slope = model.coef_[0][0]

    return slope

