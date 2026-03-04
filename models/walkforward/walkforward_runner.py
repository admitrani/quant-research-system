import pandas as pd
import numpy as np
from pathlib import Path
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from models.model_factory import get_model
from sklearn.metrics import accuracy_score, roc_auc_score
from models.metrics import compute_classification_metrics, compute_vectorized_trading_metrics, compute_degradation_slope


def load_gold_dataset():

    project_root = Path(__file__).resolve().parents[2]
    gold_path = project_root / "storage" / "gold" / "btcusdt_1h_v1.parquet"
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

def run_walkforward_for_model(X, y, df_full, windows, model_name, threshold, annualization_factor, save_results=True):

    all_results = []
    oos_returns_all = []

    for i, window in enumerate(windows):
        
        X_train, y_train, X_test, y_test = split_window(X, y, window)
        X_train_s, X_test_s, scaler = scale_window(X_train, X_test)

        model = get_model(model_name)
        model.fit(X_train_s, y_train)

        y_proba = model.predict_proba(X_test_s)[:, 1]

        positions = (y_proba > threshold).astype(int)
        future_returns = df_full.loc[X_test.index, "future_return"].values
        strategy_returns = positions * future_returns
        oos_returns_all.extend(strategy_returns)
        
        ml_metrics = compute_classification_metrics(y_test, y_proba, threshold=threshold)
        trading_metrics = compute_vectorized_trading_metrics(future_returns, y_proba, threshold=threshold, annualization_factor=annualization_factor)

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

    if save_results:
        project_root = Path(__file__).resolve().parents[2]
        results_path = project_root / "models" / "results"
        results_path.mkdir(parents=True, exist_ok=True)

        results_df.to_csv(results_path / f"walkforward_{model_name}.csv", index=False)

    return results_df, equity_curve, oos_returns_all