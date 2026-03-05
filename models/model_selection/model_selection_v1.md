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
- [0.45, 0.5, 0.55, 0.6]

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

## Selected Model

Model: Random Forest  
Depth: 6  
Threshold: 0.55  

Baseline Sharpe ≈ 0.95  
Robust Sharpe ≈ 1.38  

---

## Testing

Unit tests cover:

- walk-forward integrity
- leakage prevention
- robustness experiment
- candidate selection logic

Tests can be run with:

pytest