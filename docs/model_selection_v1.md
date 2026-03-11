# Model Candidate Selection (v1)

This module implements the final model selection process for the trading strategy.

The objective is to identify the most robust machine learning model using walk-forward validation and robustness analysis.

---

## Research Pipeline

The research pipeline is composed of three stages:

metrics → robustness → candidate selection

It can be executed via:

python -m models.research.research_pipeline

or from a specific stage:

python -m models.research.research_pipeline --stage robustness

---

## Stage 1 — Model Comparison

Script: `metrics_summary.py`

Runs a full walk-forward validation for each candidate model and computes trading metrics.

Metrics computed:

- Global Sharpe
- Mean Sharpe per window
- Sharpe stability
- Expectancy
- Final equity multiple
- Negative windows

Results saved to:

models/results/model_comparison_v1.csv

---

## Stage 2 — Robustness Analysis

Script: `robustness_runner.py`

Evaluates model sensitivity to hyperparameters.

Grid tested:

Random Forest
- max_depth: [4, 6, 8]

XGBoost
- max_depth: [3, 4, 5]

Threshold
- [0.5, 0.55, 0.6, 0.65]

Total configurations tested: 24

Results saved to:

models/results/robustness_v1.csv

---

## Stage 3 — Candidate Selection

Script: `candidate_selection.py`

Selects the final model based on:

- baseline performance
- robustness statistics
- parameter stability

A robust configuration is selected from the top 30% of Sharpe results.

Final artifact:

models/results/candidate_v1.json

Example output:


{
    "model": "rf",
    "max_depth": 6,
    "threshold": 0.55
}

---

## Selected Candidate

- **Model:** XGBoost
- **Max Depth:** 4
- **Threshold:** 0.60 (robust configuration)

### Baseline Metrics (threshold=0.5)
- Sharpe Global: 0.735
- Final Equity Multiple: 3.12x

### Robust Configuration Metrics (threshold=0.6)
- Sharpe Global: 1.497
- Final Equity Multiple: 4.71x
- Mean Sharpe Window: 1.599
- Std Sharpe Window: 1.672
- Negative Windows: 1/10

### Selection Rationale

Random Forest achieved a higher base Sharpe (0.837 vs 0.735), but XGBoost was
selected by the composite scoring function that integrates robustness metrics.
XGBoost showed better performance under threshold perturbation and fewer
negative configurations across the grid.

Source: `candidate_v1.json`, `model_comparison_v1.csv`, `robustness_v1.csv`

---

## Testing

Unit tests cover:

- walk-forward integrity
- leakage prevention
- robustness experiment
- candidate selection logic

Tests can be run with:

pytest