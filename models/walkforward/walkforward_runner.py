import pandas as pd
import numpy as np
from pathlib import Path
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from models.model_factory import get_model
from models.metrics import compute_classification_metrics, compute_vectorized_trading_metrics, compute_daily_sharpe
from models.utils import get_annualization_factor
from config.config_loader import get_gold_path


def load_gold_dataset():

    gold_path = get_gold_path()
    df = pd.read_parquet(gold_path)

    df["open_time_utc"] = pd.to_datetime(df["open_time_utc"])
    df = df.sort_values("open_time_utc")
    df.set_index("open_time_utc", inplace=True)
    return df

    
def prepare_features_and_target(df):

    feature_cols = [
        "return_1h",
        "return_3h",
        "return_12h",
        "volatility_12h",
        "ma20_distance",
        "volume_zscore",
    ]

    X = df[feature_cols].copy()
    y = df["label"].copy()

    return X, y


def generate_expanding_windows(df, initial_train_years, test_months):

    windows = []
    start_date = df.index.min()
    end_date = df.index.max()

    train_start = start_date
    train_end = train_start + relativedelta(years=initial_train_years)
    test_duration = relativedelta(months=test_months)

    while True:
        test_start = train_end
        test_end = test_start + test_duration

        if test_end > end_date:
            break

        windows.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end
        })

        # Expanding: only move train_end
        train_end = train_end + test_duration
    
    return windows


def split_window(X, y, window):

    train_mask = (X.index >= window["train_start"]) & (X.index < window["train_end"])
    test_mask = (X.index >= window["test_start"]) & (X.index < window["test_end"])

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    return X_train, y_train, X_test, y_test


def scale_window(X_train, X_test):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

def prepare_walkforward_windows(X, y, windows, df_full):

    prepared = []

    # Per-bar returns for the next bar (t+1).
    # The model is trained on a t+3 label, but the vectorized metric
    # uses t+1 returns because:
    # 1. The real backtest (Backtrader) accumulates PnL bar-by-bar,
    #    not in fixed 3-bar blocks.
    # 2. Using t+3 returns would create overlapping return windows
    #    when the model signals BUY on consecutive bars.
    # 3. t+1 is the most conservative proxy: each bar in-position
    #    captures exactly one bar of return, no double-counting.
    # The gap between vectorized Sharpe and backtest Sharpe is
    # primarily driven by transaction costs (B3), not this horizon.

    returns_1h = df_full["close_price"].pct_change().shift(-1)

    for window in windows:

        X_train, y_train, X_test, y_test = split_window(X, y, window)
        X_train_s, X_test_s, scaler = scale_window(X_train, X_test)
        bar_returns = returns_1h.loc[X_test.index].fillna(0).to_numpy()

        prepared.append({
            "X_train": X_train_s,
            "y_train": y_train,
            "X_test": X_test_s,
            "y_test": y_test,
            "future_returns": bar_returns,
            "test_dates": X_test.index,
        })

    return prepared


def run_walkforward_for_model(prepared_windows, model_name, threshold, max_depth=None, annualization_factor=None, save_results=True):

    all_results = []
    oos_returns_all = []
    oos_dates_all = []

    if annualization_factor is None:
        annualization_factor = get_annualization_factor()

    for i, window_data in enumerate(prepared_windows):
        
        X_train_s = window_data["X_train"]
        y_train = window_data["y_train"]
        X_test_s = window_data["X_test"]
        y_test = window_data["y_test"]
        test_dates = window_data["test_dates"]

        model = get_model(model_name, max_depth=max_depth)
        model.fit(X_train_s, y_train)

        y_proba = model.predict_proba(X_test_s)[:, 1]

        positions = (y_proba > threshold).astype(int)
        future_returns = window_data["future_returns"]
        strategy_returns = positions * future_returns
        oos_returns_all.extend(strategy_returns)
        oos_dates_all.extend(test_dates)
        
        ml_metrics = compute_classification_metrics(y_test, y_proba, threshold=threshold)
        trading_metrics = compute_vectorized_trading_metrics(future_returns, y_proba, threshold=threshold, annualization_factor=annualization_factor)

        trading_metrics["sharpe"] = compute_daily_sharpe(strategy_returns, test_dates)

        result_row = {
            "model": model_name,
            "window": i + 1,
            **ml_metrics,
            **trading_metrics,
        }

        all_results.append(result_row)

    results_df = pd.DataFrame(all_results)

    oos_returns_all = np.array(oos_returns_all)
    equity_curve = (1 + oos_returns_all).cumprod()
    oos_dates_all = pd.DatetimeIndex(oos_dates_all)

    if save_results:
        project_root = Path(__file__).resolve().parents[2]
        results_path = project_root / "models" / "results"
        results_path.mkdir(parents=True, exist_ok=True)

        results_df.to_csv(results_path / f"walkforward_{model_name}.csv", index=False)

    return results_df, equity_curve, oos_returns_all, oos_dates_all
