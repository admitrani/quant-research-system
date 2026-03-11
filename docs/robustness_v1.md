# 4.8.5 Structural Robustness Analysis

This module evaluates the **structural robustness** of the trading models developed in previous stages.

The objective is not to optimize performance but to verify whether the observed edge remains stable under small parameter perturbations.

---

## Models evaluated

Logistic Regression was excluded from this analysis due to significantly weaker performance in the previous module.

The robustness experiment focuses on:

- Random Forest
- XGBoost

---

## Experimental design

Two parameters were perturbed locally around their baseline values.

### Threshold

The decision threshold converts predicted probabilities into trading signals.

signal = P(y=1) > threshold

Values tested:
- 0.50
- 0.55
- 0.60
- 0.65

These values represent small perturbations around the baseline (0.5) and avoid aggressive hyperparameter tuning.

### Max Depth

Depth controls model complexity.

Random Forest:
- 4
- 6 (baseline)
- 8

XGBoost:
- 3
- 4 (baseline)
- 5

Total configurations:

3 depths × 4 thresholds × 2 models = 24 experiments

---

## Metrics

The robustness analysis focuses on **financial stability**, not classification metrics.

Per configuration metrics:

| metric | description |
|---|---|
| sharpe_global | global out-of-sample Sharpe |
| final_equity_multiple | cumulative equity growth |
| mean_sharpe_window | average walk-forward Sharpe |
| std_sharpe_window | Sharpe variability across windows |
| negative_windows | number of windows with negative Sharpe |

---

## Implementation

The experiment is implemented in:

models/robustness/robustness_runner.py

The workflow:

1. load Gold dataset
2. generate walk-forward windows
3. cache prepared windows
4. run parameter grid
5. compute performance metrics
6. export results

Results are stored in:

models/results/robustness_v1.csv

---

## Runtime optimization

Two optimizations were introduced:

### Window caching

Prepared windows store:

- scaled train/test features
- aligned future returns

This avoids recomputing splits and scaling for every configuration.

### Parallel execution

The grid is executed using:

joblib.Parallel

Each configuration runs independently.

---

## Results Summary

The selected candidate is XGBoost (depth=4, threshold=0.60) via composite scoring.
RF achieved higher base Sharpe (0.837 vs 0.735) but XGBoost showed better robustness
across the parameter grid (mean Sharpe 1.073 vs 0.729).

Note: Configurations with threshold=0.65 at shallow depths produce sharpe=0.0
and equity=1.0x — the model generates no signals above this threshold,
so no trades are executed. These appear as "neutral" in the grid.

Source: `robustness_v1.csv`

---

## Conclusion

The analysis suggests that the observed edge is **structurally stable** and not dependent on a single parameter configuration.

The selected candidate model proceeds to the next stage of the project: **model selection and validation**.
