# Walk-Forward Model Evaluation (v1)

This module implements a fully reproducible expanding walk-forward validation framework for evaluating predictive models on BTCUSDT 1H data.

The objective is to assess whether a statistically detectable edge exists under strict temporal validation, avoiding leakage and simulating realistic retraining conditions.

---

# Models Evaluated

Models are defined in `model_factory.py` and trained under identical validation conditions:

- **Logistic Regression** (linear baseline)
- **Random Forest** (non-linear ensemble)
- **XGBoost** (gradient boosting)

Hyperparameters are intentionally conservative in v1 to avoid overfitting and preserve experimental integrity. No hyperparameter tuning is performed in this version.

---

# Validation Scheme

Controlled via `v1.yaml`.

- **Scheme:** Expanding walk-forward  
- **Initial training window:** 2 years  
- **Test window:** 6 months  
- **Retraining:** Full retraining per window  
- **Total OOS windows:** 10  

### Temporal Split

Each window strictly follows:

- train: [start, train_end)
- test: [train_end, test_end)

No overlap is allowed. No future information is used in training.

### Why Expanding?

- Prevents temporal leakage  
- Simulates cumulative learning  
- Enables degradation analysis  
- Preserves historical information  

---

# Feature & Target Setup

Features originate from the Gold layer:

- `return_1h`
- `return_3h`
- `return_12h`
- `volatility_12h`
- `ma20_distance`
- `volume_zscore`

Target:

- Binary label derived from `future_return`
- Horizon defined in config

All preprocessing is performed per window:

- `StandardScaler` fitted on training data only  
- Test set transformed using training scaler  
- Strict prevention of data leakage  

---

# Metrics

Metrics are computed at two levels.

---

## Per Window Metrics

### Classification Metrics

- Accuracy  
- AUC  
- Precision  
- Recall  
- F1 Score  
- Predicted positive rate  
- True positive rate  

### Trading Metrics (Vectorized)

Signal definition:

- position = (probability > threshold)
- strategy_return = position * future_return

Computed:

- Mean return  
- Standard deviation of returns  
- Sharpe ratio (annualized dynamically)  
- Hit ratio  
- Expectancy  

Annualization factor is derived from the timeframe in `v1.yaml`. No hardcoded constants are used.

---

## Global OOS Metrics

Out-of-sample returns from all windows are concatenated to simulate continuous deployment.

Computed:

- Global Sharpe  
- Final equity multiple  
- Equity curve (concatenated OOS)  
- Degradation slope (AUC and Sharpe)  
- Number of windows with AUC < 0.5  
- Number of windows with negative Sharpe  

This avoids relying solely on per-window averages and better approximates real deployment.

---

# Results Summary (v1)

| Model    | Mean AUC | Global Sharpe | Mean Sharpe/Window | Neg. Windows | Final Equity |
|----------|----------|---------------|--------------------|-------------|-------------|
| RF       | 0.552    | 0.837         | 1.062              | 2/10        | 3.97x       |
| XGBoost  | 0.544    | 0.735         | 0.828              | 2/10        | 3.12x       |
| Logistic | 0.540    | 0.375         | 0.542              | 3/10        | 1.37x       |

Source: `model_comparison_v1.csv`, `walkforward_summary_v1.csv`

## Observations

- All models consistently outperform random (AUC > 0.5).
- Random Forest achieves the highest global Sharpe (0.837) and equity multiple (3.97x).
- XGBoost shows similar AUC but lower Sharpe variability (std 1.08 vs 1.40 for RF).
- 2-3 out of 10 windows show negative Sharpe across models.
- No catastrophic regime breakdown observed.

The detected edge is modest but structurally consistent.

---

# Reproducibility & Engineering Guarantees

- All parameters controlled via `v1.yaml`
- No hardcoded validation parameters
- Annualization dynamically derived from timeframe
- Config snapshot stored with results
- Random seeds fixed (RF, XGB)
- Strict temporal validation (no leakage)
- Pytest structural validation
- Fully executable via pipeline stage

Example:

```bash
python -m orchestration.pipeline --stage model
```

---

# Limitations (v1)

This module intentionally excludes:
- Hyperparameter tuning
- Threshold optimization
- Short positions
- Position sizing logic
- Transaction cost modeling
- Regime-specific evaluation
- Feature selection
- Ensemble strategies
- These aspects are addressed in subsequent modules.

---

# Architectural Overview

Layer separation:

YAML Config
    ↓
Model Stage (orchestration)
    ↓
Walkforward Runner (training logic)
    ↓
Metrics (pure computation)

This ensures:
- Reproducibility
- Modularity
- Config-driven experimentation
- Clean separation of concerns

---

# Methodology Note: Research vs Backtest Metrics

Research pipeline metrics are computed WITHOUT transaction costs. This is
intentional: the research pipeline evaluates predictive quality across models
under identical conditions. Since costs are equal for all models, including
them would not change the model ranking but would add implementation complexity.

The backtest pipeline applies the full cost model (0.18% commission + 0.05%
slippage + 0.3% buffer per side) with realistic execution via Backtrader.
This is the source of truth for strategy viability.

The gap between research Sharpe (~0.74 for XGB) and backtest Sharpe (-2.32)
is explained by: transaction costs, compounding with risk_fraction=1.0,
and the minimum_capital kill-switch at $10k.
